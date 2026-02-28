import time
from tqdm import tqdm

from code_dataset import create_dataset
from code_model import create_model
from code_config.parser import parse
from code_record.visualizer import Visualizer
from code_util import util
from code_network.tools.scheduler import get_num_epochs


# =========================================================
# Helpers
# =========================================================
def _is_improved(curr: float, best: float | None, mode: str = "max", min_delta: float = 0.0) -> bool:
    """
    mode="max": curr > best + min_delta
    mode="min": curr < best - min_delta
    """
    if best is None:
        return True
    if mode == "max":
        return curr > best + min_delta
    elif mode == "min":
        return curr < best - min_delta
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _get_monitor_value(monitor_key: str, losses_avg: dict, metrics_avg: dict):
    """
    监控值来源：优先 metrics，其次 losses
    """
    if monitor_key in metrics_avg:
        return float(metrics_avg[monitor_key]), "metrics"
    if monitor_key in losses_avg:
        return float(losses_avg[monitor_key]), "losses"
    return None, None


class EarlyStopping:
    """
    只负责 early stop 判断，不负责保存模型（避免和你原 save_best 重复）。
    """
    def __init__(self, monitor: str, mode: str, patience: int, min_delta: float,
                 warmup_epochs: int = 0, start_epoch: int = 1, verbose: bool = True):
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.start_epoch = int(start_epoch)
        self.verbose = bool(verbose)

        self.best = None
        self.best_epoch = None
        self.num_bad = 0

        assert self.mode in ["max", "min"], "mode must be 'max' or 'min'"

    def step(self, current_value: float, epoch: int):
        """
        Returns: should_stop(bool), improved(bool)
        """
        # warmup / start
        if epoch < self.start_epoch or epoch <= self.warmup_epochs:
            # warmup 期间也更新 best（用于显示），但不累积 bad
            if self.best is None:
                self.best = current_value
                self.best_epoch = epoch
            return False, False

        improved = _is_improved(current_value, self.best, mode=self.mode, min_delta=self.min_delta)
        if improved:
            self.best = current_value
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1

        should_stop = self.num_bad >= self.patience
        return should_stop, improved


# =========================================================
# Train
# =========================================================
def train(status_config=None, common_config=None):

    # opt >>>> config
    config, common_config = parse("train", status_config=status_config, common_config=common_config)
    val_config, _ = parse("train", status_config=status_config, save=False, val=True)

    # random seed
    util.set_random_seed(config["random_seed"])

    # dataset
    train_loader, _ = create_dataset(config)
    val_loader, val_len = create_dataset(val_config)

    # model
    model = create_model(config)
    model.setup(config)
    model.update_epoch(0)

    # visualizer
    visualizer = Visualizer(config)

    total_iters = 0
    num_epochs = get_num_epochs(config)
    start_time = time.time()

    use_html = config["record"].get("html", {}).get("use_html", False)
    use_tensorboard = config["record"].get("tensorboard", {}).get("use_tensorboard", False)

    # ------------------------
    # Save best + Early stop config (统一 monitor)
    # ------------------------
    save_cfg = config["record"].get("save_model", {})
    use_save_best = bool(save_cfg.get("save_best", False))

    es_cfg = config["record"].get("early_stop", {})
    use_early_stop = bool(es_cfg.get("enable", False))

    # monitor：优先 early_stop.monitor，否则跟随 save_model.best_metric，否则默认 ssim
    monitor_key = es_cfg.get("monitor", save_cfg.get("best_metric", "ssim"))
    mode = es_cfg.get("mode", "max")  # ssim 通常 max；若监控 L1 请设 min
    patience = int(es_cfg.get("patience", 10))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    warmup_epochs = int(es_cfg.get("warmup_epochs", 0))
    start_epoch = int(es_cfg.get("start_epoch", 1))
    verbose_es = bool(es_cfg.get("verbose", True))

    # best 值：用 None 更通用（max/min 都适用）
    best_value = None

    early_stopper = None
    if use_early_stop:
        early_stopper = EarlyStopping(
            monitor=monitor_key,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            warmup_epochs=warmup_epochs,
            start_epoch=start_epoch,
            verbose=verbose_es,
        )

        msg = (f"[EarlyStop] enabled monitor='{monitor_key}' mode='{mode}' "
               f"patience={patience} min_delta={min_delta} warmup_epochs={warmup_epochs} start_epoch={start_epoch}")
        tqdm.write(msg)
        visualizer.record_log(msg, phase="train")

    # =========================================================
    # (可选) continue_train: 在训练开始之前进行一次 validation
    # =========================================================
    if val_len > 0 and config.get("continue", {}).get("continue_train", False) is True:
        epoch = 0
        val_start_time = time.time()
        val_losses = {}
        val_metrics = {}
        model.eval()

        for data in tqdm(val_loader, desc="epoch %d/%d - val" % (epoch, num_epochs), position=1, leave=False):
            model.set_input(data)
            model.calculate_loss()
            losses = model.get_current_losses()
            model.calclulate_metric()
            metrics = model.get_current_metrics()
            val_losses = util.merge_dicts_add_values(val_losses, losses)
            val_metrics = util.merge_dicts_add_values(val_metrics, metrics)

        val_losses_avg = util.dict_divided_by_number(val_losses, len(val_loader))
        val_metrics_avg = util.dict_divided_by_number(val_metrics, len(val_loader))

        log_info_val = (
            f"Epoch {epoch}/{num_epochs} - Time: {time.time() - val_start_time:.2f}s - "
            f"val Losses: {util.dict2str(val_losses_avg)} - val Metrics: {util.dict2str(val_metrics_avg)}"
        )
        tqdm.write(log_info_val)
        visualizer.record_log(log_info_val, phase="val")

        visuals = model.get_current_visuals()
        if use_html:
            visualizer.display_on_html(visuals, data["A"]["params"]["path"], phase="val", epoch=epoch)
        if use_tensorboard:
            visualizer.display_on_tensorboard(model.get_current_visuals(), step=epoch, phase="val")
            visualizer.plot_scalars_on_tensorboard(val_losses_avg, epoch, phase="val")
            visualizer.plot_scalars_on_tensorboard(val_metrics_avg, epoch, phase="val")

        # best 保存（沿用你原思想：提升就保存 best；这里只是 epoch=0 的一次校验）
        if use_save_best:
            current, src = _get_monitor_value(monitor_key, val_losses_avg, val_metrics_avg)
            if current is not None and _is_improved(current, best_value, mode=mode, min_delta=min_delta):
                best_value = current
                tqdm.write(f"[Best] New best {monitor_key}={best_value:.6f} (src={src}) at epoch {epoch}")
                # 你原逻辑：满足 per_epoch 才保存额外快照
                per_epoch = int(save_cfg.get("per_epoch", 1))
                if per_epoch > 0 and epoch % per_epoch == 0:
                    model.save_networks(f"{epoch}_{monitor_key}_{best_value:.6f}")
                model.save_networks("best")

    # =========================================================
    # Main training loop
    # =========================================================
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", position=0):

        model.update_epoch(epoch)
        model.train()

        epoch_iter = 0
        train_losses = {}
        train_metrics = {}
        epoch_start_time = time.time()
        iter_data_time = time.time()

        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", position=1, leave=False):
            total_iters += config["dataset"]["dataloader"]["batch_size"]
            epoch_iter += config["dataset"]["dataloader"]["batch_size"]

            if epoch_iter % config["record"]["record_loss_per_iter"] == 0:
                iter_start_time = time.time()
                t_data = iter_start_time - iter_data_time

            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            model.calclulate_metric()
            metrics = model.get_current_metrics()
            train_losses = util.merge_dicts_add_values(train_losses, losses)
            train_metrics = util.merge_dicts_add_values(train_metrics, metrics)

            if epoch_iter % config["record"]["record_loss_per_iter"] == 0:
                t_comp = time.time() - iter_start_time
                log_info_train_iter = (
                    f"Epoch {epoch}/{num_epochs} - Iter {epoch_iter} - "
                    f"t_comp: {t_comp:.4f}s - t_data: {t_data:.4f}s - "
                    f"Losses: {util.dict2str(losses)} - Metrics: {util.dict2str(metrics)}"
                )
                tqdm.write(log_info_train_iter)
                visualizer.record_log(log_info_train_iter, phase="train")

            if use_html and epoch_iter % config["record"]["html"]["display_per_iter"] == 0:
                visualizer.display_on_html(model.get_current_visuals(), data["A"]["params"]["path"],
                                           phase="train", epoch=epoch, iter=epoch_iter)

            if use_tensorboard and epoch_iter % config["record"]["tensorboard"]["display_per_iter"] == 0:
                visualizer.display_on_tensorboard(model.get_current_visuals(), step=epoch_iter, phase="train")

            iter_data_time = time.time()

        # epoch avg
        t_comp = time.time() - epoch_start_time
        train_losses_avg = util.dict_divided_by_number(train_losses, len(train_loader))
        train_metrics_avg = util.dict_divided_by_number(train_metrics, len(train_loader))

        if use_tensorboard:
            visualizer.plot_scalars_on_tensorboard(train_losses_avg, epoch, phase="train")
            visualizer.plot_scalars_on_tensorboard(train_metrics_avg, epoch, phase="train")

        log_info_train_epoch = (
            f"Epoch {epoch}/{num_epochs} - Time: {t_comp:.2f}s - "
            f"Losses: {util.dict2str(train_losses_avg)} - Metrics: {util.dict2str(train_metrics_avg)}"
        )
        tqdm.write(log_info_train_epoch)
        visualizer.record_log(log_info_train_epoch, phase="train")

        model.update_learning_rate()

        if save_cfg.get("save_latest", False):
            model.save_networks("latest")

        tqdm.write("work is going on at %s" % config["work_dir"])

        # =========================================================
        # Validation + Save best + Early stop
        # =========================================================
        if val_len > 0 and epoch % config["record"]["val_per_epoch"] == 0:
            val_start_time = time.time()
            val_losses = {}
            val_metrics = {}
            model.eval()

            for data in tqdm(val_loader, desc="epoch %d/%d - val" % (epoch, num_epochs), position=1, leave=False):
                model.set_input(data)
                model.calculate_loss()
                losses = model.get_current_losses()
                model.calclulate_metric()
                metrics = model.get_current_metrics()
                val_losses = util.merge_dicts_add_values(val_losses, losses)
                val_metrics = util.merge_dicts_add_values(val_metrics, metrics)

            val_losses_avg = util.dict_divided_by_number(val_losses, len(val_loader))
            val_metrics_avg = util.dict_divided_by_number(val_metrics, len(val_loader))

            log_info_val = (
                f"Epoch {epoch}/{num_epochs} - Time: {time.time() - val_start_time:.2f}s - "
                f"val Losses: {util.dict2str(val_losses_avg)} - val Metrics: {util.dict2str(val_metrics_avg)}"
            )
            tqdm.write(log_info_val)
            visualizer.record_log(log_info_val, phase="val")

            visuals = model.get_current_visuals()
            if use_html:
                visualizer.display_on_html(visuals, data["A"]["params"]["path"], phase="val", epoch=epoch)
            if use_tensorboard:
                visualizer.display_on_tensorboard(model.get_current_visuals(), step=epoch, phase="val")
                visualizer.plot_scalars_on_tensorboard(val_losses_avg, epoch, phase="val")
                visualizer.plot_scalars_on_tensorboard(val_metrics_avg, epoch, phase="val")

            # -------------------------
            # Save best (你原本逻辑：提升就保存 best)
            # 这里改为 mode-aware + 可从 metrics 或 losses 取
            # -------------------------
            current, src = _get_monitor_value(monitor_key, val_losses_avg, val_metrics_avg)
            if current is None:
                msg = (f"[Best/EarlyStop] monitor='{monitor_key}' not found. "
                       f"Available metrics={list(val_metrics_avg.keys())}, losses={list(val_losses_avg.keys())}")
                tqdm.write(msg)
                visualizer.record_log(msg, phase="val")
            else:
                # save best
                if use_save_best and _is_improved(current, best_value, mode=mode, min_delta=min_delta):
                    best_value = current
                    tqdm.write(f"[Best] New best {monitor_key}={best_value:.6f} (src={src}) at epoch {epoch}")

                    per_epoch = int(save_cfg.get("per_epoch", 1))
                    if per_epoch > 0 and epoch % per_epoch == 0:
                        model.save_networks(f"{epoch}_{monitor_key}_{best_value:.6f}")
                    model.save_networks("best")

                # -------------------------
                # Early stop (只判断，不保存，避免重复)
                # -------------------------
                if use_early_stop and early_stopper is not None:
                    should_stop, improved_es = early_stopper.step(current, epoch)
                    if verbose_es:
                        msg = (f"[EarlyStop] epoch={epoch} monitor={monitor_key} value={current:.6f} "
                               f"best={early_stopper.best:.6f} best_epoch={early_stopper.best_epoch} "
                               f"bad={early_stopper.num_bad}/{early_stopper.patience} "
                               f"improved={improved_es} stop={should_stop}")
                        tqdm.write(msg)
                        visualizer.record_log(msg, phase="val")

                    if should_stop:
                        msg = (f"[EarlyStop] Stop training at epoch {epoch}. "
                               f"Best {monitor_key}={early_stopper.best:.6f} @ epoch {early_stopper.best_epoch}")
                        tqdm.write(msg)
                        visualizer.record_log(msg, phase="train")
                        break  # break epoch loop

        # 若触发 early stop，跳出外层 epoch loop
        if use_early_stop and early_stopper is not None and early_stopper.num_bad >= early_stopper.patience:
            break

    # =========================================================
    # End
    # =========================================================
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = f"Total time: {total_time // 3600}h {(total_time % 3600) // 60}m {total_time % 60:.2f}s"
    tqdm.write(total_time_str)
    visualizer.record_log(total_time_str, phase="train")

    return common_config


if __name__ == "__main__":
    train()
