import json
import os
import re
from collections import deque

# ==========================================
# 1. 경로 및 파라미터
# ==========================================
ROI_PATH = "roi/seats.json"
DETECTION_DIR = "json_results"

IOU_THRESHOLD = 0.03      # Seat-Person IoU threshold
WINDOW_SIZE = 5           # Temporal window
MIN_OCC_FRAMES = 2        # 최소 점유 프레임 수

# ==========================================
# 2. 유틸 함수
# ==========================================

def bbox_to_list(b):
    return [b["x1"], b["y1"], b["x2"], b["y2"]]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def is_occupied(roi_bbox, person_boxes):
    """IoU + foot-point 기반 점유 판단"""
    for p in person_boxes:
        # IoU
        if iou(roi_bbox, p) >= IOU_THRESHOLD:
            return True

        # Foot-point (bbox 하단 중앙)
        foot_x = (p[0] + p[2]) / 2
        foot_y = p[3]
        if roi_bbox[0] <= foot_x <= roi_bbox[2] and roi_bbox[1] <= foot_y <= roi_bbox[3]:
            return True
    return False

def extract_frame_idx(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0

# ==========================================
# 3. 메인 로직
# ==========================================

def main():
    # ---------- seats.json 로딩 ----------
    if not os.path.exists(ROI_PATH):
        print(f"❌ {ROI_PATH} 없음")
        return

    with open(ROI_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tables = []
    seats = []

    for obj in data:
        if "bbox" not in obj:
            continue

        label = obj.get("label", "").lower()
        bbox = bbox_to_list(obj["bbox"])

        if label == "table":
            tables.append({
                "id": f"T{len(tables)+1}",
                "bbox": bbox
            })
        elif label == "seat":
            seats.append({
                "bbox": bbox
            })

    print(f"✅ Loaded Tables: {len(tables)}")
    print(f"✅ Loaded Seats : {len(seats)}")

    if len(tables) == 0:
        print("❌ 테이블 로드 실패")
        return

    # ---------- Seat → Table 매핑 ----------
    table_seats = {t["id"]: [] for t in tables}

    for s in seats:
        sx = (s["bbox"][0] + s["bbox"][2]) / 2
        sy = (s["bbox"][1] + s["bbox"][3]) / 2

        closest = min(
            tables,
            key=lambda t: (
                (sx - (t["bbox"][0] + t["bbox"][2]) / 2) ** 2 +
                (sy - (t["bbox"][1] + t["bbox"][3]) / 2) ** 2
            )
        )
        table_seats[closest["id"]].append(s["bbox"])

    # ---------- Temporal buffer ----------
    history = {
        t["id"]: deque(maxlen=WINDOW_SIZE)
        for t in tables
    }

    # ---------- Detection JSON ----------
    files = sorted(
        [f for f in os.listdir(DETECTION_DIR) if f.endswith(".json")],
        key=extract_frame_idx
    )

    print(f"📂 Total Frames: {len(files)}")
    print("=== Table Occupancy (Seat-based) ===")

    for fname in files:
        with open(os.path.join(DETECTION_DIR, fname), "r", encoding="utf-8") as f:
            det = json.load(f)

        persons = [
            bbox_to_list(d["bbox"])
            for d in det.get("detections", [])
            if d.get("class") == "person"
        ]

        occupied = 0
        print(f"\n[{fname}] persons={len(persons)}")

        for t in tables:
            # 테이블 직접 점유 OR 소속 의자 점유
            table_occ = is_occupied(t["bbox"], persons)
            seat_occ = any(
                is_occupied(seat_bbox, persons)
                for seat_bbox in table_seats[t["id"]]
            )

            occ = table_occ or seat_occ
            history[t["id"]].append(1 if occ else 0)

            final_occ = sum(history[t["id"]]) >= MIN_OCC_FRAMES
            status = "Occupied" if final_occ else "Free"

            print(f"  {t['id']}: {status}")

            if final_occ:
                occupied += 1

        print(f"➡️ Occupancy: {occupied}/{len(tables)}")

# ==========================================
if __name__ == "__main__":
    main()
