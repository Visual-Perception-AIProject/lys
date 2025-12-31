import json
import os
import re

# ==========================================
# 1. 경로 및 파라미터
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_PATH = os.path.join(BASE_DIR, "roi", "seats.json")
DETECTION_DIR = os.path.join(BASE_DIR, "json_results")

IOU_THRESHOLD = 0.03

FPS = 30                  # 영상 FPS (필요 시 조정)
OCCUPY_SECONDS = 3        # "몇 초 이상 앉아 있으면 점유"
OCCUPY_FRAMES = FPS * OCCUPY_SECONDS

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
    """IoU + foot-point 기반"""
    for p in person_boxes:
        if iou(roi_bbox, p) >= IOU_THRESHOLD:
            return True

        # foot-point
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

    tables = []
    table_seats = {}

    for t in data:
        tid = t["id"]
        tables.append({
            "id": tid,
            "bbox": bbox_to_list(t["bbox"])
        })
        table_seats[tid] = [
            bbox_to_list(s["bbox"])
            for s in t.get("seats", [])
        ]

    print(f"✅ Loaded Tables: {len(tables)}")
    print(f"✅ Loaded Seats : {sum(len(v) for v in table_seats.values())}")

    if not tables:
        print("❌ 테이블 로드 실패")
        return

    # ---------- 시간축 상태 초기화 ----------
    table_states = {
        t["id"]: {
            "occupied_frames": 0,
            "status": "empty"   # empty | candidate | occupied
        }
        for t in tables
    }

    # ---------- Detection JSON ----------
    files = sorted(
        [f for f in os.listdir(DETECTION_DIR) if f.endswith(".json")],
        key=extract_frame_idx
    )

    print(f"📂 Total Frames: {len(files)}")
    print("=== Table Occupancy (Temporal-based) ===")

    for fname in files:
        with open(os.path.join(DETECTION_DIR, fname), "r", encoding="utf-8") as f:
            det = json.load(f)

        persons = [
            bbox_to_list(d["bbox"])
            for d in det.get("detections", [])
            if d.get("class") == "person"
        ]

        print(f"\n[{fname}] persons={len(persons)}")
        occupied_cnt = 0

        for t in tables:
            tid = t["id"]
            state = table_states[tid]

            table_occ = is_occupied(t["bbox"], persons)
            seat_occ = any(
                is_occupied(seat_bbox, persons)
                for seat_bbox in table_seats.get(tid, [])
            )

            overlapped = table_occ or seat_occ

            if overlapped:
                state["occupied_frames"] += 1

                if state["occupied_frames"] >= OCCUPY_FRAMES:
                    state["status"] = "occupied"
                else:
                    state["status"] = "candidate"
            else:
                state["occupied_frames"] = 0
                state["status"] = "empty"

            if state["status"] == "occupied":
                occupied_cnt += 1

            print(
                f"  {tid}: {state['status']} "
                f"({state['occupied_frames']}/{OCCUPY_FRAMES})"
            )

        print(f"➡️ Occupancy: {occupied_cnt}/{len(tables)}")

# ==========================================
if __name__ == "__main__":
    main()
