import math
import os
import torch
import random
import gc
from torch import optim, nn
import torch.nn.functional as F

from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS
from data.mnist_dataloader import get_sage_mnist_loader
from src.monitoring.health_tracker import ManifoldHealthTracker
from src.utils.growth_hormone import GrowthHormone

# Reference to the maturation stages defined in the architecture
STAGE_ORDER = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Elder"]
EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')


def run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args, governor, loader=None):
    """
    SAGE TRAINING ENGINE: Handles staged maturation, Hebbian grounding,
    and topological pruning with distinct Latent and Structural metrics.

    Surgical Fix: Implements Hinge-Loss Pressure to prevent negative manifold dissipation.
    Full Restoration: Includes Centroid Addresses, Health Tracking, Warmup, and Hardware Purge.
    """

    # --- HARDWARE DETECTION ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs("checkpoints", exist_ok=True)

    # --- 1. THE SURGICAL SUTURE (Conditional Freezing) ---
    # We freeze the entire container and then surgically unlock only the active stage.
    for param in sage_container.parameters():
        param.requires_grad = False

    # UNLOCK Graph parameters for Contiguous Gradient Block updates (The Manifold)
    for param in sage_container.graph.parameters():
        param.requires_grad = True

    stage_key = stage_name.capitalize()
    target_arg_stage = args.stage.capitalize()

    if stage_key in sage_container.stages:
        target_stage = sage_container.stages[stage_key]

        if stage_key == target_arg_stage:
            print(f"[*] ACTIVE TRAINING: Unlocking {stage_key} gradients.")
            target_stage.train()
            for param in target_stage.parameters():
                param.requires_grad = True

            # Enable gradient checkpointing for memory efficiency in deep stages
            if hasattr(target_stage, 'layers'):
                for layer in target_stage.layers:
                    if hasattr(layer, 'gradient_checkpointing'):
                        layer.gradient_checkpointing = True
        else:
            print(f"[*] SUTURE MODE: {stage_key} is frozen. Handshake only.")
            target_stage.eval()
    else:
        print(f"WARNING: Stage {stage_key} not found in container.")
        return

    # Ensure frontend is always trainable to maintain the handshake
    for param in frontend.parameters():
        param.requires_grad = True

    frontend.to(device)
    sage_container.to(device)

    # --- 2. OPTIMIZER & SCHEDULER HARDENING ---
    hparams = STAGE_HYPERPARAMS[stage_key]

    # The Graph receives its own parameter group to maintain breakout momentum
    optimizer = optim.AdamW([
        {
            'params': [p for p in target_stage.parameters() if p.requires_grad],
            'lr': hparams["learning_rate"]
        },
        {
            'params': [p for p in frontend.parameters() if p.requires_grad],
            'lr': hparams["learning_rate"]
        },
        {
            'params': [p for p in sage_container.graph.parameters() if p.requires_grad],
            'lr': hparams["learning_rate"]
        }
    ], weight_decay=hparams.get("weight_decay", 0.01))

    # Realization of the Linear Warmup Scheduler (Surgical Fix for Teen stage)
    warmup_steps = hparams.get("scheduler", {}).get("warmup_steps", 150)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps)))

    criterion = nn.CrossEntropyLoss()

    # --- 3. TRACKER STATE & HANDSHAKE UPDATE ---
    health_tracker = ManifoldHealthTracker(sage_container.graph)
    stage_idx = STAGE_ORDER.index(stage_key)
    sage_container.current_stage_idx = stage_idx

    # Update plasticity window for the specific stage
    sage_container.update_plasticity_window(cumulative=True)

    # PERSISTENT GROWTH CONTROLLER: Initialized outside the epoch loop to track novelty pass-to-pass
    # The floor is explicitly set here to avoid UnboundLocalError during conditional checks
    current_floor = 0.99 if stage_key == "Teen" else hparams.get("growth_confidence_floor", 0.35)
    gh = GrowthHormone(
        floor=current_floor,
        ceiling=hparams.get("confidence_ceiling", 0.99)
    )

    # DYNAMIC LOADER SELECTION
    if loader is None:
        if args.data == "mnist":
            loader = get_sage_mnist_loader(stage_name.lower(), sage_container.graph, train=True, device=device)
        elif args.data == "imagenet":
            from data.imagenet_dataloader import get_sage_imagenet_loader
            loader = get_sage_imagenet_loader(stage_name.lower(), sage_container.graph, train=True, device=device)
        else:
            loader = get_sage_mnist_loader(stage_name.lower(), sage_container.graph, train=True, device=device)

    print(f"\n=== SAGE SURGICAL TRAINING: {stage_name.upper()} ({device}) ===")

    # --- MAIN TRAINING LOOP ---
    for epoch in range(args.epochs):
        gh.reset_batch()  # Re-allow label births for the new epoch pass
        health_tracker.check_health(stage_name, epoch)
        frontend.train()
        sage_container.graph.train()

        if stage_key == target_arg_stage:
            sage_container.train()
        else:
            sage_container.eval()

        loss_total, batch_count = 0.0, 0
        target_var = hparams.get("max_structural_density", 0.40)
        p_weight = hparams.get("pressure_weight", 0.2)

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            # --- SAGE GRADIENT SUTURE ---
            # 1. Pull the live matrix for gradient flow
            current_g_matrix = sage_container.graph.get_graph_embedding_matrix()

            # 2. Derive variance directly from the live tensor
            live_variance = torch.var(current_g_matrix)

            # 3. REPAIR: HINGE LOSS (ReLU) - Pushes nodes apart if they cluster too tightly
            pressure_loss_tensor = torch.relu(torch.tensor(target_var, device=device) - live_variance)
            latent_pressure = pressure_loss_tensor.item()

            # 4. Structural Density Calculation (Real-world topology check)
            node_count = len(sage_container.graph.nodes)
            total_edges = sum(len(n.connections) for n in sage_container.graph.nodes.values())
            structural_density = total_edges / max(1, node_count * 100)

            x = batch['input_ids'].to(device, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True) if 'labels' in batch else None

            # Centroid Address handshake for grounding alignment
            c_addresses = batch.get('centroid_addresses')
            if c_addresses is not None:
                c_addresses = [c.to(device, non_blocking=True) for c in c_addresses]

            # Forward pass using the matrix that is now part of the grad chain
            logits, telemetry = frontend(
                x,
                sage_container,
                graph_matrix=current_g_matrix
            )

            winner_indices = telemetry.get('winner_node_ids')

            if y is not None and logits is not None:
                valid_mask = (y < logits.size(-1))
                if valid_mask.any():
                    # 1. TASK LOSS
                    task_loss = criterion(logits[valid_mask], y[valid_mask])

                    # 2. PRESSURE LOSS (The SAGE Breakout Fix)
                    weighted_pressure_loss = pressure_loss_tensor * p_weight
                    total_loss = task_loss + weighted_pressure_loss

                    if stage_key == target_arg_stage:
                        total_loss.backward()

                        # Secure checkpointing every 100 batches
                        if batch_count % 100 == 0:
                            governor.secure_save(sage_container.graph,
                                                 metadata={"batch": batch_count, "pressure": latent_pressure})

                        all_trainable = []
                        for group in optimizer.param_groups:
                            all_trainable.extend(group['params'])

                        torch.nn.utils.clip_grad_norm_(all_trainable, hparams.get("gradient_clip", 1.0))
                        optimizer.step()
                        scheduler.step()  # Advance the warmup scheduler

                    loss_total += total_loss.item()

            if batch_count % 10 == 0:
                mem_str = ""
                if device.type == "mps":
                    mem_used = torch.mps.current_allocated_memory() / 1024 ** 2
                    mem_str = f" | Mem: {mem_used:.1f}MB"
                elif device.type == "cuda":
                    mem_used = torch.cuda.memory_allocated() / 1024 ** 2
                    mem_str = f" | Mem: {mem_used:.1f}MB"

                print(
                    f"[PROBE] Batch {batch_count} | Conf: {telemetry.get('confidence', 0):.4f} | Press: {latent_pressure:.6f} | Density: {structural_density:.4f}{mem_str}")

            # --- MATURATION & EXPANSION (Growth Logic) ---
            if stage_key in ["Teen", "Adult", "Elder"]:
                # Ensure the object exists (safety check)
                if gh is not None:
                    print(f"DEBUG: Attempting GH Call for {stage_key}...")
                    telemetry['text'] = batch.get('raw_texts', ["latent_concept"])[0]
                    new_node_id = gh.maybe_create_node(telemetry=telemetry, graph=sage_container.graph)

                    if new_node_id is not None:
                        print(f"\n>>> [!] {stage_key.upper()} GROWTH: Seeded Node {new_node_id}")
                else:
                    print(f"!!! CRITICAL FAILURE: GrowthHormone object is NONE for stage {stage_key}")

            # --- HEBBIAN WIRING & ADAPTIVE PRUNING ---
            target_threshold = hparams.get("confidence_ceiling", 0.45)
            should_update = (stage_key == target_arg_stage) and (winner_indices is not None)

            if should_update:
                num_nodes = len(sage_container.graph.node_order)
                trace = telemetry.get('trace')

                # Restore Trace Padding logic for manifold size mismatch
                if trace is None:
                    trace = torch.ones((len(winner_indices), num_nodes), device=device)
                else:
                    trace = trace.to(device).float()
                    if trace.dim() == 3: trace = trace.squeeze(0)
                    if trace.size(0) != len(winner_indices): trace = trace[:len(winner_indices), :]

                    if trace.size(1) < num_nodes:
                        pad = torch.ones((trace.size(0), num_nodes - trace.size(1)), device=device)
                        trace = torch.cat([trace, pad], dim=1)
                    else:
                        trace = trace[:, :num_nodes]

                sage_container.graph.update_stage_aware_hebbian(
                    stage_key=stage_key,
                    tightness=latent_pressure,
                    attention_map=trace,
                    batch_indices=winner_indices,
                    stage_plasticity=hparams["plasticity_scale"],
                    threshold=target_threshold
                )

                # --- INTRA-BATCH ADAPTIVE PRUNING (Relief Valve) ---
                if structural_density > hparams.get("max_structural_density", 0.40):
                    # Sampling winners + random nodes for pruning consideration
                    sample_size = max(10, int(len(sage_container.graph.node_order) * 0.05))
                    random_sample = random.sample(sage_container.graph.node_order,
                                                  min(len(sage_container.graph.node_order), sample_size))
                    target_prune_list = list(set(winner_indices + [int(r) for r in random_sample if str(r).isdigit()]))

                    sage_container.graph.adaptive_prune(
                        tightness=structural_density,
                        max_tightness=hparams.get("max_structural_density", 0.40),
                        prune_fraction=hparams.get("prune_fraction", 0.1),
                        nodes_to_consider=target_prune_list,
                        min_connections=1
                    )

            batch_count += 1

        # --- SYNAPTIC SLEEP CYCLE (EndOfEpoch) ---
        # 1. Update gradient-based manifold positions
        sage_container.graph.apply_governed_gradient_update(lr=hparams["learning_rate"])
        print(f"[*] Epoch {epoch} complete. Initiating Synaptic Sleep Cycle...")

        # 2. Global Edge Pruning (Crystallization to break 0.99 Confidence lock)
        sleep_threshold = 0.20 if stage_key == "Teen" else 0.08
        sage_container.graph.apply_edge_threshold(min_weight=sleep_threshold)

        # 3. Global Anchor Re-Alignment
        print("[!] Re-aligning Manifold Anchors...", end="")
        for node_id in list(sage_container.graph.nodes.keys()):
            sage_container.graph.update_local_centroid(node_id)
        print(" Done.")

        avg_loss = loss_total / max(batch_count, 1)
        analytics.capture_snapshot(stage_name, stage_idx, epoch, avg_loss, telemetry, sage_container.graph)

        print(f"\n[EPOCH {epoch + 1} SUMMARY]")
        print(f" > Pressure (Latent): {latent_pressure:.6f}")
        print(f" > Density (Struct):  {structural_density:.6f}")
        print(f" > Global Loss:       {avg_loss:.6f}")
        print("-" * 45)

    # --- FINAL PERSISTENCE & HARDWARE PURGE ---
    if stage_key == target_arg_stage:
        sage_container.save_agnostic_stage(stage_key)
        governor.secure_save(sage_container.graph)
        torch.save(frontend.state_dict(), f"checkpoints/FRONTEND_{stage_key}.pth")

    analytics.save_stage_report(stage_name)
    health_tracker.check_health(stage_name, "FINAL")

    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[+] Stage {stage_name} Complete. Hardware Purged.")