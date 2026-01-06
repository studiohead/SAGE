import math
import os
import torch
import random
from torch import optim, nn

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
    for param in sage_container.parameters():
        param.requires_grad = False

    # UNLOCK Graph parameters for Contiguous Gradient Block updates
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

    # Ensure frontend is always trainable
    for param in frontend.parameters():
        param.requires_grad = True

    frontend.to(device)
    sage_container.to(device)

    # --- 2. OPTIMIZER & SPEED HARDENING ---
    hparams = STAGE_HYPERPARAMS[stage_key]

    # The Graph receives a 10x Learning Rate 'Kick' to maintain breakout momentum
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
            'lr': hparams["learning_rate"] * 1
        }
    ], weight_decay=hparams.get("weight_decay", 0.01))

    criterion = nn.CrossEntropyLoss()

    # --- 3. TRACKER STATE & HANDSHAKE UPDATE ---
    health_tracker = ManifoldHealthTracker(sage_container.graph)
    stage_idx = STAGE_ORDER.index(stage_key)
    sage_container.current_stage_idx = stage_idx

    sage_container.update_plasticity_window(cumulative=True)

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
        health_tracker.check_health(stage_name, epoch)
        frontend.train()
        sage_container.graph.train()

        if stage_key == target_arg_stage:
            sage_container.train()
        else:
            sage_container.eval()

        loss_total, batch_count = 0.0, 0

        # Pull targets from your real config
        target_var = hparams.get("max_manifold_tightness", 0.20)
        p_weight = hparams.get("pressure_weight", 0.6)

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            # --- SAGE GRADIENT SUTURE (REPAIRED) ---
            # 1. Pull the live matrix
            current_g_matrix = sage_container.graph.get_graph_embedding_matrix()

            # 2. Derive variance directly from the live tensor
            live_variance = torch.var(current_g_matrix)

            # 3. REPAIR: HINGE LOSS (ReLU)
            # This ensures that if live_variance > target_var, pressure is EXACTLY 0.0
            # It prevents the negative values and the 'Yo-Yo' pull-back.
            pressure_loss_tensor = torch.relu(torch.tensor(target_var, device=device) - live_variance)

            # 4. For logging, we use the scalar value of the CLAMPED tensor
            latent_pressure = pressure_loss_tensor.item()
            # ---------------------------------------

            node_count = len(sage_container.graph.nodes)
            total_edges = sum(len(n.connections) for n in sage_container.graph.nodes.values())
            structural_density = total_edges / max(1, node_count * 100)

            x = batch['input_ids'].to(device, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True) if 'labels' in batch else None

            c_addresses = batch.get('centroid_addresses')
            if c_addresses is not None:
                c_addresses = [c.to(device, non_blocking=True) for c in c_addresses]

            # Forward pass using the matrix that is now part of the grad chain
            logits, telemetry = frontend(
                x,
                sage_container,
                graph_matrix=current_g_matrix,
                centroid_addresses=c_addresses
            )

            winner_indices = telemetry.get('winner_node_ids')

            if y is not None and logits is not None:
                valid_mask = (y < logits.size(-1))
                if valid_mask.any():
                    # 1. TASK LOSS
                    task_loss = criterion(logits[valid_mask], y[valid_mask])

                    # 2. PRESSURE LOSS (The SAGE Breakout Fix)
                    # We multiply the Hinge-clamped tensor by the weight
                    weighted_pressure_loss = pressure_loss_tensor * p_weight
                    total_loss = task_loss + weighted_pressure_loss

                    if stage_key == target_arg_stage:
                        total_loss.backward()

                        if batch_count % 100 == 0:
                            governor.secure_save(sage_container.graph,
                                                 metadata={"batch": batch_count, "pressure": latent_pressure})

                        all_trainable = []
                        for group in optimizer.param_groups:
                            all_trainable.extend(group['params'])

                        torch.nn.utils.clip_grad_norm_(all_trainable, hparams.get("gradient_clip", 1.0))
                        optimizer.step()

                    loss_total += total_loss.item()

            if batch_count % 10 == 0:
                mem_str = ""
                if device.type == "mps":
                    mem_used = torch.mps.current_allocated_memory() / 1024 ** 2
                    mem_str = f" | Mem: {mem_used:.1f}MB"
                elif device.type == "cuda":
                    mem_used = torch.cuda.memory_allocated() / 1024 ** 2
                    mem_str = f" | Mem: {mem_used:.1f}MB"

                if winner_indices:
                    first_winner_id = str(winner_indices[0])
                    label = sage_container.graph.nodes[first_winner_id].label
                    print(f"[VERIFY] Winner {first_winner_id} Label: {label}")

                print(
                    f"[PROBE] Batch {batch_count} | Winners: {winner_indices[:3] if winner_indices else 'None'} | "
                    f"Pressure: {latent_pressure:.6f} | Density: {structural_density:.4f}{mem_str}")

            # --- MATURATION & EXPANSION ---
            current_stage_idx = STAGE_ORDER.index(stage_key)
            teen_stage_idx = STAGE_ORDER.index("Teen")

            if current_stage_idx >= teen_stage_idx:
                gh = GrowthHormone(floor=hparams.get("growth_confidence_floor", 0.51),
                                   ceiling=hparams.get("confidence_threshold", 0.88))
                if 'raw_texts' in batch and len(batch['raw_texts']) > 0:
                    telemetry['text'] = batch['raw_texts'][0]
                elif y is not None:
                    telemetry['text'] = f"node_label_{y[0].item()}"

                new_node_id = gh.maybe_create_node(telemetry=telemetry, graph=sage_container.graph)
                if new_node_id is not None:
                    print(f"\n>>> [!] {stage_key.upper()} GROWTH: Seeded Node {new_node_id}")

            # --- HEBBIAN WIRING ---
            target_threshold = hparams.get("confidence_threshold", 0.45)
            conf = telemetry.get('confidence', 0.0)
            should_update = (stage_key == target_arg_stage) and (stage_key == "Infant" or conf >= target_threshold)

            if should_update and winner_indices is not None:
                trace = telemetry.get('trace')
                if trace is None: trace = torch.ones((1, len(winner_indices)), device=device)

                print(f" [!] Hebbian Wiring...", end="", flush=True)
                # tightness is now correctly logged as the clamped value
                sage_container.graph.update_stage_aware_hebbian(
                    stage_key=stage_key, tightness=latent_pressure, attention_map=trace,
                    batch_indices=winner_indices, stage_plasticity=hparams["plasticity_scale"],
                    threshold=target_threshold
                )
                print(" Done.")

                # ADAPTIVE PRUNING
                if structural_density > hparams.get("max_manifold_tightness", 0.20):
                    sample_size = max(10, int(len(sage_container.graph.node_order) * 0.05))
                    random_sample = random.sample(sage_container.graph.node_order, sample_size)
                    target_prune_list = list(set(winner_indices + [int(r) for r in random_sample if str(r).isdigit()]))
                    sage_container.graph.adaptive_prune(
                        tightness=structural_density, max_tightness=hparams.get("max_manifold_tightness", 0.20),
                        prune_fraction=hparams.get("prune_fraction", 0.6), nodes_to_consider=target_prune_list
                    )

            batch_count += 1
        # Alpha 0.2 means we move nodes 20% closer to the origin in one shot
        sage_container.graph.apply_origin_attractor(alpha=0.2, target_drift=0.5)
        print(f"[*] Epoch {epoch} complete. Initiating Synaptic Sleep Cycle...")
        sage_container.graph.apply_edge_threshold(min_weight=0.08)

        avg_loss = loss_total / max(batch_count, 1)
        analytics.capture_snapshot(stage_name, stage_idx, epoch, avg_loss, telemetry, sage_container.graph)

        print(f"\n[EPOCH {epoch + 1} SUMMARY]")
        print(f" > Pressure (Latent Delta): {latent_pressure:.6f}")
        print(f" > Density (Struct):       {structural_density:.6f}")
        print(f" > Global Loss:            {avg_loss:.6f}")
        print("-" * 45)

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