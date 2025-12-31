import json
import os
import re
from collections import deque

# ==========================================
# 1. 경로 및 파라미터
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_PATH = os.path.join(BASE_DIR, "roi", "seats.json")
DETECTION_DIR = os.path.join(BASE_DIR, "json_results")

IOU_THRESHOLD = 0.03
WINDOW_SIZE = 5
MIN_OCC_FRAMES = 2

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
    for p in person_boxes:
        if iou(roi_bbox, p) >= IOU_THRESHOLD:
            return True

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
    print(f"📌 ROI_PATH: {ROI_PATH}")

    if not os.path.exists(ROI_PATH):
        print("❌ seats.json 없음")
        return

    with open(ROI_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    print(f"📌 JSON type: {type(data)}")
    print(f"📌 Tables in JSON: {len(data)}")

    tables = []
    table_seats = {}

    # ---------- Table / Seat 파싱 ----------
    for t in data:
        table_id = t.get("id")
        table_bbox = bbox_to_list(t["bbox"])

        tables.append({
            "id": table_id,
            "bbox": table_bbox
        })

        table_seats[table_id] = [
            bbox_to_list(s["bbox"])
            for s in t.get("seats", [])
        ]

    print(f"✅ Loaded Tables: {len(tables)}")
    print(f"✅ Loaded Seats : {sum(len(v) for v in table_seats.values())}")

    if len(tables) == 0:
        print("❌ 테이블 로드 실패")
        return

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
            tid = t["id"]

            table_occ = is_occupied(t["bbox"], persons)
            seat_occ = any(
                is_occupied(seat_bbox, persons)
                for seat_bbox in table_seats.get(tid, [])
            )

            occ = table_occ or seat_occ
            history[tid].append(1 if occ else 0)

            final_occ = sum(history[tid]) >= MIN_OCC_FRAMES
            status = "Occupied" if final_occ else "Free"

            print(f"  {tid}: {status}")

            if final_occ:
                occupied += 1

        print(f"➡️ Occupancy: {occupied}/{len(tables)}")

# ==========================================
if __name__ == "__main__":
    main()
