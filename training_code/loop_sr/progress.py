"""Console progress only; full metrics continue to be written to JSONL."""
from tqdm import tqdm


class TrainingProgress:
    def __init__(self, settings, step, total):
        self.enabled = settings.get("progress", True)
        self.json_console = settings.get("json_console", False)
        self.bar = tqdm(total=total, initial=step, desc="Train", unit="step",
                        dynamic_ncols=True, ascii=True, mininterval=0.5,
                        disable=not self.enabled)

    def update(self, loss, rate, vram=None):
        values = {"loss": f"{loss:.5f}", "lr": f"{rate:.2e}"}
        if vram is not None:
            values["VRAM"] = f"{vram:.2f}G"
        self.bar.set_postfix(values, refresh=False)
        self.bar.update(1)

    def write(self, message):
        tqdm.write(message)

    def validation(self, step, report, loops):
        scores = [f"{mode}: PSNR={values[f'loop_{loops}']['psnr_rgb']:.3f}"
                  for mode, values in report.items()]
        self.write(f"Validation step {step} | " + " | ".join(scores))

    def close(self):
        self.bar.close()
