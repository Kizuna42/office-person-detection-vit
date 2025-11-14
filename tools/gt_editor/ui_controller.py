"""UIコントローラーモジュール: キーボードとマウスイベントの処理"""

from collections.abc import Callable
import logging

import cv2

logger = logging.getLogger(__name__)


class UIController:
    """UIイベントコントローラー"""

    def __init__(
        self,
        window_name: str,
        on_mouse_callback: Callable[[int, int, int, int, int], None],
    ):
        """初期化

        Args:
            window_name: ウィンドウ名
            on_mouse_callback: マウスイベントコールバック関数
        """
        self.window_name = window_name
        self.on_mouse_callback = on_mouse_callback
        self.id_input_mode = False
        self.id_input_buffer = ""

    def setup(self) -> None:
        """UIをセットアップ"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse_wrapper)

    def _on_mouse_wrapper(self, event, x, y, flags, param):
        """マウスイベントのラッパー（OpenCVのコールバック形式に合わせる）"""
        self.on_mouse_callback(event, x, y, flags, param)

    def process_key(
        self,
        key: int,
        on_quit: Callable[[], None],
        on_save: Callable[[], None],
        on_add_track: Callable[[], None],
        on_delete_point: Callable[[], None],
        on_match_id: Callable[[], None],
        on_change_track_id: Callable[[int], None],
        on_start_id_input: Callable[[], None],
        on_frame_change: Callable[[int], None],
    ) -> bool:
        """キー入力を処理

        Args:
            key: キーコード
            on_quit: 終了コールバック
            on_save: 保存コールバック
            on_add_track: トラック追加コールバック
            on_delete_point: ポイント削除コールバック
            on_match_id: IDマッチングコールバック
            on_change_track_id: トラックID変更コールバック
            on_start_id_input: ID入力モード開始コールバック
            on_frame_change: フレーム変更コールバック（delta: -1 or 1）

        Returns:
            終了する場合True
        """
        # 終了
        if key == ord("q") or (key == 27 and not self.id_input_mode):  # 'q' or ESC
            return True

        # 保存
        if key == ord("s"):
            on_save()

        # トラック追加
        elif key == ord("a"):
            on_add_track()

        # ポイント削除
        elif key == ord("d"):
            on_delete_point()

        # IDマッチング
        elif key == ord("m"):
            on_match_id()

        # ID入力モード開始
        elif key == ord("i"):
            on_start_id_input()

        # ID入力モード中
        elif self.id_input_mode:
            if ord("0") <= key <= ord("9"):  # 数字キー
                digit = key - ord("0")
                if len(self.id_input_buffer) < 2:  # 最大2桁
                    self.id_input_buffer += str(digit)
                    logger.info(f"ID入力: {self.id_input_buffer}")
            elif key == 13 or key == 10:  # Enterキー
                if self.id_input_buffer:
                    try:
                        new_id = int(self.id_input_buffer)
                        if 1 <= new_id <= 30:
                            on_change_track_id(new_id)
                        else:
                            logger.warning(f"IDは1-30の範囲で指定してください: {new_id}")
                    except ValueError:
                        logger.warning(f"無効なID: {self.id_input_buffer}")
                self.id_input_mode = False
                self.id_input_buffer = ""
            elif key == 27:  # ESCキー（キャンセル）
                self.id_input_mode = False
                self.id_input_buffer = ""
                logger.info("ID入力モードをキャンセルしました")

        # クイックID変更（1-9）
        elif ord("1") <= key <= ord("9"):
            new_id = key - ord("0")
            on_change_track_id(new_id)

        # 矢印キー
        else:
            # 左矢印
            if key in (81, 2, 0x250000, 65361):
                on_frame_change(-1)
            # 右矢印
            elif key in (83, 3, 0x270000, 65363):
                on_frame_change(1)

        return False

    def get_id_input_state(self) -> tuple[bool, str]:
        """ID入力モードの状態を取得

        Returns:
            (id_input_mode, id_input_buffer) のタプル
        """
        return self.id_input_mode, self.id_input_buffer
