import logging
import sys
from pathlib import Path


class LoggerSetup:
    """
    Cấu hình logging tập trung cho toàn bộ project.

    Thiết kế:
    - File log: DEBUG trở lên.
    - Console: INFO trở lên.
    - Tự xóa handler cũ để tránh duplicate log trong Colab/Kaggle/Jupyter.
    """

    @staticmethod
    def initialize(
        log_file: Path,
        clear_old_logs: bool = True,
        force_reset: bool = True,
    ) -> None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger()

        if force_reset:
            for handler in root_logger.handlers[:]:
                handler.flush()
                handler.close()
                root_logger.removeHandler(handler)

        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_mode = "w" if clear_old_logs else "a"

        file_handler = logging.FileHandler(
            filename=log_file,
            mode=file_mode,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Giảm bớt log nhiễu từ thư viện ngoài.
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        logging.getLogger("LoggerSetup").info(
            "Logger đã sẵn sàng. File log: %s", log_file
        )

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)
