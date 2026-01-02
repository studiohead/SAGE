import os
import torch
from torch import optim, nn

from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS
from data.mnist_dataloader import get_sage_mnist_loader
from src.monitoring.health_tracker import ManifoldHealthTracker
from src.utils.growth_hormone import GrowthHormone

STAGE_ORDER = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Elder"]
EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')


def run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args, governor, loader=None):
    # Detect Hardware: Priority MPS (Mac) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs("checkpoints", exist_ok=True)

    # 1. THE SURGICAL SUTURE (Conditional Freezing)
    for param in sage_container.parameters():
        param.requires_grad = False

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

    for param in frontend.parameters():
        param.requires_grad = True

    frontend.to(device)
    sage_container.to(device)

    # 2. Optimizer & Speed Hardening
    hparams = STAGE_HYPERPARAMS[stage_key]
    trainable_params = [p for p in sage_container.parameters() if p.requires_grad] + list(frontend.parameters())

    optimizer = optim.AdamW(trainable_params, lr=hparams["learning_rate"],
                            weight_decay=hparams.get("weight_decay", 0.01))
    criterion = nn.CrossEntropyLoss()

    # 3. Tracker State
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

    for epoch in range(args.epochs):
        health_tracker.check_health(stage_name, epoch)

        frontend.train()
        if stage_key == target_arg_stage:
            sage_container.train()
        else:
            sage_container.eval()

        loss_total, batch_count, total_gamma = 0.0, 0, 0.0
        telemetry = {}
        active_nodes = []

        static_graph = None
        if stage_key != "Infant":
            static_graph = sage_container.graph.get_graph_embedding_matrix()

        active_ids = list(sage_container.graph.node_order)

        for batch in loader:
            x = batch['input_ids'].to(device, non_blocking=True)
            batch_indices = batch.get('node_indices')
            y = batch['labels'].to(device, non_blocking=True) if 'labels' in batch else None

            # PROBE: Heartbeat
            if batch_count % 5 == 0:
                mem_str = ""
                if device.type == "mps":
                    mem_used = torch.mps.current_allocated_memory() / 1024 ** 2
                    mem_str = f" | MPS Mem: {mem_used:.1f}MB"

                display_indices = batch_indices[:5] if (
                        batch_indices is not None and hasattr(batch_indices, '__getitem__')) else "N/A"
                print(f"[PROBE] Batch {batch_count} | Winners: {display_indices}{mem_str}")

            c_addresses = batch.get('centroid_addresses')
            if c_addresses is not None:
                c_addresses = [c.to(device, non_blocking=True) for c in c_addresses]

            if hasattr(x, 'shape') and len(x.shape) > 1 and x.shape[1] != EMBED_DIM and args.data == "text":
                padded = torch.zeros((x.size(0), EMBED_DIM), device=device)
                padded[:, :min(x.size(1), EMBED_DIM)] = x[:, :min(x.size(1), EMBED_DIM)]
                x = padded

            current_g_matrix = static_graph if static_graph is not None else sage_container.graph.get_graph_embedding_matrix()

            optimizer.zero_grad(set_to_none=True)

            logits, telemetry = frontend(
                x,
                sage_container,
                graph_matrix=current_g_matrix,
                centroid_addresses=c_addresses
            )

            # --- MATURATION & EXPANSION: TEEN AND UP ---
            current_stage_idx = STAGE_ORDER.index(stage_key)
            teen_stage_idx = STAGE_ORDER.index("Teen")

            if current_stage_idx >= teen_stage_idx:
                # EFFICIENT CONFIG LEVERAGE: No hardcoding.
                gh_floor = hparams.get("growth_confidence_floor", 0.5)
                gh_ceiling = hparams.get("confidence_threshold", 0.7)

                growth_hormone = GrowthHormone(floor=gh_floor, ceiling=gh_ceiling)

                if 'raw_texts' in batch and len(batch['raw_texts']) > 0:
                    telemetry['text'] = batch['raw_texts'][0]
                elif y is not None:
                    telemetry['text'] = f"node_label_{y[0].item()}"
                else:
                    telemetry['text'] = "unknown_sensory_node"

                # PATH A: CREATION
                new_node_id = growth_hormone.maybe_create_node(telemetry=telemetry, graph=sage_container.graph)

                if new_node_id is not None:
                    active_nodes.append(new_node_id)
                    active_ids.append(new_node_id)
                    assigned_label = sage_container.graph.nodes[str(new_node_id)].label
                    print(f"\n>>> [!] {stage_key.upper()} GROWTH: Node {new_node_id} ({assigned_label})")

                # PATH B: MATURATION
                else:
                    conf = telemetry.get('confidence', 0.0)
                    winner_id = telemetry.get('winner_node_id')
                    if conf >= hparams["confidence_threshold"] and winner_id is not None:
                        current_node = sage_container.graph.nodes.get(str(winner_id))
                        if current_node and current_node.label != telemetry['text']:
                            governor.rebrand_node(sage_container.graph, winner_id, telemetry['text'])

            # --- MASKED SUPERVISED LOSS ---
            if y is not None and logits is not None:
                valid_mask = (y < logits.size(-1))
                if valid_mask.any():
                    loss = criterion(logits[valid_mask], y[valid_mask])
                    if stage_key == target_arg_stage:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, hparams.get("gradient_clip", 1.0))
                        optimizer.step()
                    loss_total += loss.item()

            # --- UNIVERSAL HEBBIAN GROUNDING ---
            target_threshold = hparams.get("confidence_threshold", 0.7)
            conf = telemetry.get('confidence', 0.0)
            is_infant = (stage_key == "Infant")

            should_update = (stage_key == target_arg_stage) and (is_infant or conf >= target_threshold)

            if should_update and batch_indices is not None:
                trace = telemetry.get('trace')
                if trace is None or (torch.is_tensor(trace) and trace.sum() == 0):
                    trace = torch.ones((1, len(batch_indices)), device=device)

                print(f" [!] Hebbian Wiring (Conf: {conf:.2f})...", end="", flush=True)
                sage_container.graph.update_stage_aware_hebbian(
                    stage_key=stage_key,
                    tightness=1.0,
                    attention_map=trace,
                    batch_indices=batch_indices,
                    stage_plasticity=hparams["plasticity_scale"],
                    threshold=0.0 if is_infant else target_threshold
                )
                print(" Done.")

            current_gamma = telemetry.get('gamma_divergence', torch.tensor(0.0, device=device))
            total_gamma += current_gamma.item() if not torch.isnan(current_gamma) else 0.0
            batch_count += 1

            if batch_count % 50 == 0 and device.type == "mps":
                torch.mps.empty_cache()

        # --- ACTIVE NODE PRUNING POST-BATCH (PRESERVED LOGIC) ---
        if active_nodes:
            variances = []
            for node_id in active_nodes:
                node_key = str(node_id)
                if node_key not in sage_container.graph.nodes:
                    continue
                node = sage_container.graph.nodes[node_key]
                if node.is_tombstoned or node.alignment_score <= 0.0:
                    continue
                diffs = []
                for other in sage_container.graph.nodes.values():
                    if other.is_tombstoned or other.alignment_score <= 0.0:
                        continue
                    diffs.append(torch.sum((other.embedding - node.embedding.to(other.embedding.device)) ** 2))
                if diffs:
                    variances.append(torch.mean(torch.stack(diffs)))

            # 1. Calculate the actual variance of the graph manifold
            # current_var = sage_container.graph.compute_manifold_variance()

            # 2. Call pruner leveraging config hyperparameters
            # sage_container.graph.adaptive_prune(
            #     tightness=current_var,
            #     max_tightness=hparams.get("max_manifold_tightness", 0.05),
            #     prune_fraction=hparams.get("prune_fraction", 0.1),
            #     nodes_to_consider=active_ids
            # )

        avg_loss = loss_total / max(batch_count, 1)
        avg_gamma = total_gamma / max(batch_count, 1)

        snapshot = analytics.capture_snapshot(stage_name, stage_idx, epoch, avg_loss, telemetry, sage_container.graph)
        node_count = len(sage_container.graph.nodes)
        edge_count = sum(len(node.connections) for node in sage_container.graph.nodes.values())
        avg_degree = edge_count / max(node_count, 1)
        degrees = [len(node.connections) for node in sage_container.graph.nodes.values()]

        print(f" > Max Node Degree: {max(degrees) if degrees else 0}")
        print(f" > Min Node Degree: {min(degrees) if degrees else 0}")

        print(f"\n[EPOCH {epoch + 1} COMPLETE]")
        print(f" > Status:      {'TRAINING' if stage_key == target_arg_stage else 'SUTURED'}")
        print(f" > Loss:        {avg_loss:.6f}")
        print(f" > Gamma (Γ):   {avg_gamma:.6f}")
        print(f" > Variance:    {snapshot['metrics']['manifold_variance']:.6f}")
        print(f" > Nodes:       {node_count}")
        print(f" > Avg Node Degree: {avg_degree:.2f}")
        print(f" > Active Edges: {edge_count}")
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
    print(f"[+] Stage {stage_name} Complete. Hardware Purged.")