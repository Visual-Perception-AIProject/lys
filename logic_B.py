import json
import os
import re
from collections import defaultdict

# ==========================================
# 1. 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_PATH = os.path.join(BASE_DIR, "roi2", "seats.json")
DETECTION_DIR = os.path.join(BASE_DIR, "json_results2")

# ==========================================
# 2. 파라미터 (조절 가능)
# ==========================================
MIN_STAY_FRAMES = 10   # 이 프레임 이상 머물면 점유 (≈ 몇 초)
RESET_ON_MISS = True  # 중간에 벗어나면 초기화

# ==========================================
# 3. 유틸 함수
# ==========================================

def bbox_to_list(b):
    return [b["x1"], b["y1"], b["x2"], b["y2"]]

def extract_frame_idx(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0

def foot_point(person_bbox):
    x1, y1, x2, y2 = person_bbox
    return ((x1 + x2) / 2, y2)

def point_in_bbox(px, py, bbox):
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]

# ==========================================
# 4. 메인 로직 B
# ==========================================

def main():
    print("📌 ROI_PATH:", ROI_PATH)

    if not os.path.exists(ROI_PATH):
        print("❌ seats.json 없음")
        return

    # ---------- seats.json 로딩 ----------
    with open(ROI_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # ---------- Seat bbox만 분리 ----------
    seats = []
    for i, obj in enumerate(data):
        if obj.get("label", "").lower() == "seat":
            seats.append({
                "id": f"S{i+1}",
                "bbox": bbox_to_list(obj["bbox"])
            })

    print(f"✅ Loaded Seats: {len(seats)}")

    if len(seats) == 0:
        print("❌ Seat 없음")
        return

    # ---------- 좌석별 체류 프레임 ----------
    stay_counter = defaultdict(int)
    occupied_seats = set()

    # ---------- detection json ----------
    files = sorted(
        [f for f in os.listdir(DETECTION_DIR) if f.endswith(".json")],
        key=extract_frame_idx
    )

    print(f"📂 Total Frames: {len(files)}")
    print("=== Seat Occupancy (Logic B: Foot + Time) ===")

    for fname in files:
        with open(os.path.join(DETECTION_DIR, fname), "r", encoding="utf-8") as f:
            det = json.load(f)

        persons = [
            bbox_to_list(d["bbox"])
            for d in det.get("detections", [])
            if d.get("class") == "person"
        ]

        hit_seats = set()

        # ---------- 발 위치 검사 ----------
        for p in persons:
            fx, fy = foot_point(p)

            for seat in seats:
                if point_in_bbox(fx, fy, seat["bbox"]):
                    stay_counter[seat["id"]] += 1
                    hit_seats.add(seat["id"])

        # ---------- 벗어난 좌석 초기화 ----------
        if RESET_ON_MISS:
            for seat in seats:
                sid = seat["id"]
                if sid not in hit_seats:
                    stay_counter[sid] = 0

        # ---------- 점유 판정 ----------
        for sid, cnt in stay_counter.items():
            if cnt >= MIN_STAY_FRAMES:
                occupied_seats.add(sid)

        print(f"[{fname}] Occupied Seats: {len(occupied_seats)}")

    print("\n✅ 최종 점유 좌석:")
    for sid in sorted(occupied_seats):
        print(" ", sid)

# ==========================================
if __name__ == "__main__":
    main()
