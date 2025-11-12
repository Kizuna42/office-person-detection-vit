"""Zone classification module for the office person detection system."""

import logging

logger = logging.getLogger(__name__)


class ZoneClassifier:
    """ゾーン分類クラス

    点in多角形アルゴリズム（Ray Casting）を使用して、
    フロアマップ上の座標がどのゾーンに属するかを判定する。

    Attributes:
        zones: ゾーン定義のリスト
    """

    def __init__(self, zones: list[dict], allow_overlap: bool = True):
        """ZoneClassifierを初期化する

        Args:
            zones: ゾーン定義のリスト
                各ゾーンは以下の形式:
                {
                    'id': 'zone_a',
                    'name': '会議室エリア',
                    'polygon': [[x1, y1], [x2, y2], ...],
                    'priority': 1  # オプション
                }
            allow_overlap: Trueの場合は重複ゾーンを許可し、検出されたすべてのゾーンIDを返す。
                            Falseの場合は優先順位ロジックに基づき1つのゾーンのみを返す。

        Raises:
            ValueError: ゾーン定義が不正な場合
        """
        self.allow_overlap = allow_overlap
        self.zones = self._validate_zones(zones)
        logger.info(
            "ZoneClassifierを初期化しました。ゾーン数: %d, allow_overlap=%s",
            len(self.zones),
            self.allow_overlap,
        )

    def _validate_zones(self, zones: list[dict]) -> list[dict]:
        """ゾーン定義を検証する

        Args:
            zones: ゾーン定義のリスト

        Returns:
            検証済みのゾーン定義

        Raises:
            ValueError: ゾーン定義が不正な場合
        """
        if not isinstance(zones, list):
            raise ValueError("zonesはリストである必要があります。")

        validated_zones = []
        zone_ids = set()

        for i, zone in enumerate(zones):
            if not isinstance(zone, dict):
                raise ValueError(f"zones[{i}]は辞書である必要があります。")

            # 必須フィールドのチェック
            if "id" not in zone:
                raise ValueError(f"zones[{i}]には'id'が必要です。")

            zone_id = zone["id"]
            if zone_id in zone_ids:
                raise ValueError(f"重複したゾーンID: {zone_id}")
            zone_ids.add(zone_id)

            if "polygon" not in zone:
                raise ValueError(f"zones[{i}]には'polygon'が必要です。")

            polygon = zone["polygon"]
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(f"zones[{i}].polygonは少なくとも3つの頂点が必要です。")

            # 頂点座標の検証
            validated_polygon = []
            for j, point in enumerate(polygon):
                if not isinstance(point, list | tuple) or len(point) != 2:
                    raise ValueError(f"zones[{i}].polygon[{j}]は[x, y]形式である必要があります。")

                try:
                    x, y = float(point[0]), float(point[1])
                    validated_polygon.append((x, y))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"zones[{i}].polygon[{j}]の座標は数値である必要があります。") from e

            validated_zone = {
                "id": zone_id,
                "name": zone.get("name", zone_id),
                "polygon": validated_polygon,
                "priority": None,
                "_order": i,
            }

            if "priority" in zone and zone["priority"] is not None:
                try:
                    priority_value = float(zone["priority"])
                except (TypeError, ValueError) as e:
                    raise ValueError(f"zones[{i}].priority は数値である必要があります。") from e
                validated_zone["priority"] = priority_value
            validated_zones.append(validated_zone)

            logger.debug(f"ゾーン検証完了: {zone_id} ({len(validated_polygon)}頂点)")

        return validated_zones

    def classify(self, floor_point: tuple[float, float]) -> list[str]:
        """座標が属するゾーンIDのリストを返す

        複数のゾーンに重複する場合、すべての該当ゾーンIDを返す。
        どのゾーンにも属さない場合は空リストを返す。

        Args:
            floor_point: フロアマップ座標 (x, y)

        Returns:
            所属するゾーンIDのリスト
        """
        matched_zones: list[dict] = []

        for zone in self.zones:
            if self._point_in_polygon(floor_point, zone["polygon"]):
                matched_zones.append(zone)

        if not matched_zones:
            logger.debug(f"座標 {floor_point} はどのゾーンにも属しません。")
            return []

        if self.allow_overlap:
            zone_ids = [zone["id"] for zone in matched_zones]
        else:
            selected_zone = min(
                matched_zones,
                key=lambda z: (
                    z["priority"] if z["priority"] is not None else float("inf"),
                    z["_order"],
                ),
            )
            zone_ids = [selected_zone["id"]]

        logger.debug(f"座標 {floor_point} はゾーン {zone_ids} に属します。")
        return zone_ids

    def classify_batch(self, floor_points: list[tuple[float, float]]) -> list[list[str]]:
        """複数の座標をバッチ分類する

        Args:
            floor_points: フロアマップ座標のリスト

        Returns:
            各座標に対応するゾーンIDリストのリスト
        """
        return [self.classify(point) for point in floor_points]

    def _point_in_polygon(self, point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        """点が多角形内にあるか判定する（Ray Casting Algorithm）

        点から右方向に水平線を引き、多角形の辺との交点数を数える。
        交点数が奇数なら内部、偶数なら外部と判定する。

        Args:
            point: 判定する点 (x, y)
            polygon: 多角形の頂点リスト [(x1, y1), (x2, y2), ...]

        Returns:
            点が多角形内にある場合True、外部の場合False
        """
        x, y = point
        n = len(polygon)
        inside = False

        # 多角形の各辺について判定
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]

            # 点のy座標が辺のy座標範囲内にあるかチェック
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                # 辺が垂直でない場合
                if p1y != p2y:
                    # 交点のx座標を計算
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                # 辺が垂直、または点が交点より左側にある場合
                if p1x == p2x or x <= xinters:
                    inside = not inside

            p1x, p1y = p2x, p2y

        return inside

    def get_zone_info(self, zone_id: str) -> dict | None:
        """ゾーンIDから詳細情報を取得する

        Args:
            zone_id: ゾーンID

        Returns:
            ゾーン情報の辞書、存在しない場合はNone
        """
        for zone in self.zones:
            if zone["id"] == zone_id:
                return zone
        return None

    def get_all_zone_ids(self) -> list[str]:
        """すべてのゾーンIDを取得する

        Returns:
            ゾーンIDのリスト
        """
        return [zone["id"] for zone in self.zones]

    def get_zone_count(self) -> int:
        """ゾーンの総数を取得する

        Returns:
            ゾーン数
        """
        return len(self.zones)

    def classify_with_unclassified(self, floor_point: tuple[float, float]) -> list[str]:
        """座標を分類し、未分類の場合は"unclassified"を返す

        Args:
            floor_point: フロアマップ座標 (x, y)

        Returns:
            所属するゾーンIDのリスト（未分類の場合は["unclassified"]）
        """
        zone_ids = self.classify(floor_point)

        if not zone_ids:
            return ["unclassified"]

        return zone_ids
