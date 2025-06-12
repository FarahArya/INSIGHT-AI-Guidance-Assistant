import cv2, json, time, numpy as np, os, sys
from ultralytics import YOLO, solutions

# ────────────── REAL HEIGHTS ───────────────
REAL_HEIGHTS = {
    "Person": 1.70,
    "Sneakers": 0.12,
    "Chair": 0.90,
    "Other Shoes": 0.10,
    "Hat": 0.18,
    "Car": 1.50,
    "Lamp": 1.20,
    "Glasses": 0.05,
    "Bottle": 0.25,
    "Desk": 0.75,
    "Cup": 0.10,
    "Street Lights": 5.00,
    "Cabinet/shelf": 1.80,
    "Handbag/Satchel": 0.35,
    "Bracelet": 0.02,
    "Plate": 0.02,
    "Picture/Frame": 0.40,
    "Helmet": 0.25,
    "Book": 0.25,
    "Gloves": 0.20,
    "Storage box": 0.50,
    "Boat": 2.00,
    "Leather Shoes": 0.10,
    "Flower": 0.30,
    "Bench": 0.45,
    "Potted Plant": 0.60,
    "Bowl/Basin": 0.10,
    "Flag": 1.50,
    "Pillow": 0.50,
    "Boots": 0.30,
    "Vase": 0.35,
    "Microphone": 0.20,
    "Necklace": 0.40,
    "Ring": 0.02,
    "SUV": 1.80,
    "Wine Glass": 0.15,
    "Belt": 0.03,
    "Monitor/TV": 0.50,
    "Backpack": 0.45,
    "Umbrella": 0.90,
    "Traffic Light": 3.00,
    "Speaker": 0.40,
    "Watch": 0.03,
    "Tie": 0.40,
    "Trash bin Can": 0.80,
    "Slippers": 0.08,
    "Bicycle": 1.10,
    "Stool": 0.45,
    "Barrel/bucket": 0.50,
    "Van": 2.20,
    "Couch": 0.80,
    "Sandals": 0.05,
    "Basket": 0.40,
    "Drum": 0.60,
    "Pen/Pencil": 0.15,
    "Bus": 3.00,
    "Wild Bird": 0.30,
    "High Heels": 0.12,
    "Motorcycle": 1.30,
    "Guitar": 1.00,
    "Carpet": 0.02,
    "Cell Phone": 0.15,
    "Bread": 0.10,
    "Camera": 0.15,
    "Canned": 0.12,
    "Truck": 3.50,
    "Traffic cone": 0.75,
    "Cymbal": 0.05,
    "Lifesaver": 0.50,
    "Towel": 0.02,
    "Stuffed Toy": 0.30,
    "Candle": 0.20,
    "Sailboat": 10.00,
    "Laptop": 0.03,
    "Awning": 2.00,
    "Bed": 0.60,
    "Faucet": 0.30,
    "Tent": 1.80,
    "Horse": 1.60,
    "Mirror": 1.20,
    "Power outlet": 0.10,
    "Sink": 0.85,
    "Apple": 0.08,
    "Air Conditioner": 0.40,
    "Knife": 0.25,
    "Hockey Stick": 1.50,
    "Paddle": 1.20,
    "Pickup Truck": 1.90,
    "Fork": 0.18,
    "Traffic Sign": 2.00,
    "Balloon": 0.30,
    "Tripod": 1.50,
    "Dog": 0.60,
    "Spoon": 0.15,
    "Clock": 0.30,
    "Pot": 0.25,
    "Cow": 1.40,
    "Cake": 0.15,
    "Dining Table": 0.75,
    "Sheep": 0.90,
    "Hanger": 0.40,
    "Blackboard/Whiteboard": 1.20,
    "Napkin": 0.02,
    "Other Fish": 0.30,
    "Orange/Tangerine": 0.08,
    "Toiletry": 0.15,
    "Keyboard": 0.03,
    "Tomato": 0.08,
    "Lantern": 0.40,
    "Machinery Vehicle": 3.00,
    "Fan": 0.40,
    "Green Vegetables": 0.15,
    "Banana": 0.18,
    "Baseball Glove": 0.25,
    "Airplane": 10.00,
    "Mouse": 0.03,
    "Train": 4.00,
    "Pumpkin": 0.30,
    "Soccer": 0.22,
    "Skiboard": 1.70,
    "Luggage": 0.70,
    "Nightstand": 0.60,
    "Tea pot": 0.20,
    "Telephone": 0.20,
    "Trolley": 1.00,
    "Head Phone": 0.20,
    "Sports Car": 1.20,
    "Stop Sign": 2.10,
    "Dessert": 0.08,
    "Scooter": 1.00,
    "Stroller": 1.00,
    "Crane": 15.00,
    "Remote": 0.15,
    "Refrigerator": 1.80,
    "Oven": 0.85,
    "Lemon": 0.08,
    "Duck": 0.40,
    "Baseball Bat": 0.85,
    "Surveillance Camera": 0.25,
    "Cat": 0.30,
    "Jug": 0.30,
    "Broccoli": 0.15,
    "Piano": 1.00,
    "Pizza": 0.04,
    "Elephant": 3.00,
    "Skateboard": 0.12,
    "Surfboard": 1.80,
    "Gun": 0.30,
    "Skating and Skiing shoes": 0.15,
    "Gas stove": 0.85,
    "Donut": 0.03,
    "Bow Tie": 0.10,
    "Carrot": 0.15,
    "Toilet": 0.80,
    "Kite": 0.80,
    "Strawberry": 0.03,
    "Other Balls": 0.20,
    "Shovel": 1.50,
    "Pepper": 0.12,
    "Computer Box": 0.45,
    "Toilet Paper": 0.10,
    "Cleaning Products": 0.25,
    "Chopsticks": 0.20,
    "Microwave": 0.30,
    "Pigeon": 0.30,
    "Baseball": 0.07,
    "Cutting/chopping Board": 0.02,
    "Coffee Table": 0.45,
    "Side Table": 0.60,
    "Scissors": 0.20,
    "Marker": 0.15,
    "Pie": 0.08,
    "Ladder": 3.00,
    "Snowboard": 1.60,
    "Cookies": 0.02,
    "Radiator": 0.60,
    "Fire Hydrant": 1.00,
    "Basketball": 0.24,
    "Zebra": 1.30,
    "Grape": 0.02,
    "Giraffe": 5.00,
    "Potato": 0.10,
    "Sausage": 0.15,
    "Tricycle": 0.70,
    "Violin": 0.60,
    "Egg": 0.06,
    "Fire Extinguisher": 0.60,
    "Candy": 0.02,
    "Fire Truck": 3.20,
    "Billiards": 0.06,
    "Converter": 0.12,
    "Bathtub": 0.55,
    "Wheelchair": 0.95,
    "Golf Club": 1.00,
    "Briefcase": 0.35,
    "Cucumber": 0.18,
    "Cigar/Cigarette": 0.10,
    "Paint Brush": 0.20,
    "Pear": 0.10,
    "Heavy Truck": 4.00,
    "Hamburger": 0.10,
    "Extractor": 0.60,
    "Extension Cord": 0.03,
    "Tong": 0.30,
    "Tennis Racket": 0.68,
    "Folder": 0.30,
    "American Football": 0.15,
    "earphone": 0.05,
    "Mask": 0.20,
    "Kettle": 0.25,
    "Tennis": 0.06,
    "Ship": 15.00,
    "Swing": 2.50,
    "Coffee Machine": 0.40,
    "Slide": 3.00,
    "Carriage": 1.50,
    "Onion": 0.10,
    "Green beans": 0.15,
    "Projector": 0.30,
    "Frisbee": 0.03,
    "Washing Machine/Drying Machine": 0.90,
    "Chicken": 0.40,
    "Printer": 0.40,
    "Watermelon": 0.25,
    "Saxophone": 0.70,
    "Tissue": 0.12,
    "Toothbrush": 0.18,
    "Ice cream": 0.15,
    "Hot-air balloon": 25.00,
    "Cello": 1.20,
    "French Fries": 0.12,
    "Scale": 0.10,
    "Trophy": 0.40,
    "Cabbage": 0.15,
    "Hot dog": 0.15,
    "Blender": 0.35,
    "Peach": 0.08,
    "Rice": 0.03,
    "Wallet/Purse": 0.15,
    "Volleyball": 0.21,
    "Deer": 1.20,
    "Goose": 0.80,
    "Tape": 0.10,
    "Tablet": 0.25,
    "Cosmetics": 0.15,
    "Trumpet": 0.60,
    "Pineapple": 0.25,
    "Golf Ball": 0.04,
    "Ambulance": 2.50,
    "Parking meter": 1.20,
    "Mango": 0.12,
    "Key": 0.08,
    "Hurdle": 1.00,
    "Fishing Rod": 2.00,
    "Medal": 0.10,
    "Flute": 0.40,
    "Brush": 0.20,
    "Penguin": 0.70,
    "Megaphone": 0.30,
    "Corn": 0.20,
    "Lettuce": 0.20,
    "Garlic": 0.05,
    "Swan": 0.90,
    "Helicopter": 4.50,
    "Green Onion": 0.30,
    "Sandwich": 0.10,
    "Nuts": 0.03,
    "Speed Limit Sign": 2.00,
    "Induction Cooker": 0.10,
    "Broom": 1.40,
    "Trombone": 0.80,
    "Plum": 0.05,
    "Rickshaw": 1.50,
    "Goldfish": 0.12,
    "Kiwi fruit": 0.07,
    "Router/modem": 0.15,
    "Poker Card": 0.01,
    "Toaster": 0.25,
    "Shrimp": 0.03,
    "Sushi": 0.03,
    "Cheese": 0.08,
    "Notepaper": 0.02,
    "Cherry": 0.02,
    "Pliers": 0.20,
    "CD": 0.01,
    "Pasta": 0.05,
    "Hammer": 0.30,
    "Cue": 1.50,
    "Avocado": 0.12,
    "Hami melon": 0.20,
    "Flask": 0.25,
    "Mushroom": 0.10,
    "Screwdriver": 0.20,
    "Soap": 0.08,
    "Recorder": 0.30,
    "Bear": 2.00,
    "Eggplant": 0.20,
    "Board Eraser": 0.05,
    "Coconut": 0.20,
    "Tape Measure/Ruler": 0.30,
    "Pig": 0.90,
    "Showerhead": 0.20,
    "Globe": 0.40,
    "Chips": 0.05,
    "Steak": 0.05,
    "Crosswalk Sign": 2.50,
    "Stapler": 0.08,
    "Camel": 1.80,
    "Formula 1": 1.00,
    "Pomegranate": 0.12,
    "Dishwasher": 0.85,
    "Crab": 0.15,
    "Hoverboard": 0.20,
    "Meatball": 0.05,
    "Rice Cooker": 0.30,
    "Tuba": 0.90,
    "Calculator": 0.15,
    "Papaya": 0.20,
    "Antelope": 1.40,
    "Parrot": 0.35,
    "Seal": 1.20,
    "Butterfly": 0.05,
    "Dumbbell": 0.15,
    "Donkey": 1.20,
    "Lion": 1.20,
    "Urinal": 0.70,
    "Dolphin": 2.00,
    "Electric Drill": 0.25,
    "Hair Dryer": 0.20,
    "Egg tart": 0.05,
    "Jellyfish": 0.30,
    "Treadmill": 1.20,
    "Lighter": 0.08,
    "Grapefruit": 0.12,
    "Game board": 0.05,
    "Mop": 1.50,
    "Radish": 0.15,
    "Baozi": 0.08,
    "Target": 1.80,
    "French": 0.20,
    "Spring Rolls": 0.05,
    "Monkey": 0.60,
    "Rabbit": 0.30,
    "Pencil Case": 0.05,
    "Yak": 1.70,
    "Red Cabbage": 0.20,
    "Binoculars": 0.15,
    "Asparagus": 0.25,
    "Barbell": 0.25,
    "Scallop": 0.08,
    "Noddles": 0.05,
    "Comb": 0.15,
    "Dumpling": 0.05,
    "Oyster": 0.08,
    "Table Tennis paddle": 0.25,
    "Cosmetics Brush/Eyeliner Pencil": 0.15,
    "Chainsaw": 0.45,
    "Eraser": 0.03,
    "Lobster": 0.40,
    "Durian": 0.30,
    "Okra": 0.10,
    "Lipstick": 0.08,
    "Cosmetics Mirror": 0.15,
    "Curling": 0.90,
    "Table Tennis": 0.04
}


# ────────────── CONFIGURATION ───────────────
MODEL_DIR = os.getenv('MODEL_PATH', './models/yolo11n_object365.pt')
LABELS = YOLO(MODEL_DIR).names
CONF_THRES = 0.45
NEAR_THRESH_METRES = 6
TRIGGER_FILE = os.getenv('TRIGGER_PATH', '../../shared/trigger.txt')
FEEDBACK_FILE = os.getenv('FEEDBACK_PATH', '../../shared/feedback.json')
IMG_SZ = 416

# Load focal length from calibration file if available
try:
    with open("calib_cam.json") as f:
        FOCAL_PX = json.load(f)["focal_px"]
        print(f"[INFO] Using calibrated focal = {FOCAL_PX}", file=sys.stderr)
except (FileNotFoundError, KeyError):
    FOCAL_PX = 600
    print("[WARN] calib_cam.json missing – using default.", file=sys.stderr)

# ────────────── GUI Safety Guard for Headless ───────────────
if not os.getenv("DISPLAY"):
    def _nogui(*a, **k): return None
    cv2.namedWindow = cv2.imshow = cv2.destroyAllWindows = _nogui
    cv2.setMouseCallback = _nogui
    cv2.waitKey = lambda *a, **k: -1
    print("[INFO] HighGUI disabled.", file=sys.stderr)

# ────────────── Performance Flags ───────────────
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setUseOptimized(True)

# ────────────── Distance Estimation Fallback ───────────────
def estimate_distance_fallback(box, img_h):
    x1, y1, x2, y2 = box.xyxy[0]
    h_px = float(y2 - y1)
    label = LABELS[int(box.cls[0])]
    real_h = REAL_HEIGHTS.get(label, None)
    if real_h:
        return (real_h * FOCAL_PX) / h_px
    return (img_h / h_px) * 0.5

# ────────────── Bounding Box Helper ───────────────
def get_box_centroid(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# ────────────── Natural Language Response ───────────────
def create_response_text(nearby_objects):
    if len(nearby_objects) == 1:
        dist, label = nearby_objects[0]
        return f"There is a {label} approximately {dist:.0f} metres ahead."

    nearby_objects.sort(key=lambda x: x[0])
    if len(nearby_objects) == 2:
        obj1, obj2 = nearby_objects
        return f"There is a {obj1[1]} approximately {obj1[0]:.0f} metres ahead, and a {obj2[1]} at {obj2[0]:.0f} metres."

    parts = []
    for i, (dist, label) in enumerate(nearby_objects):
        if i == 0:
            parts.append(f"There is a {label} at {dist:.0f} metres")
        elif i == len(nearby_objects) - 1:
            parts.append(f"and a {label} at {dist:.0f} metres ahead")
        else:
            parts.append(f"a {label} at {dist:.0f} metres")
    return ", ".join(parts) + "."

# ────────────── Interactive Distance Calculator ───────────────
class InteractiveDistanceCalculator:
    def __init__(self, distance_calc):
        self.distance_calc = distance_calc

    def process_frame(self, frame):
        results = self.distance_calc(frame)
        if hasattr(results, 'boxes') and results.boxes is not None:
            return results, results.boxes
        det = model(frame, imgsz=IMG_SZ, conf=CONF_THRES)[0]
        return det, det.boxes

    def get_distances_between_objects(self, boxes):
        distances, centroids = [], []
        for box in boxes:
            centroid = get_box_centroid(box)
            label = LABELS[int(box.cls[0])]
            centroids.append((centroid, label, box))

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                p1, l1, _ = centroids[i]
                p2, l2, _ = centroids[j]
                pd = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                est = pd / 100
                distances.append({'objects': (l1, l2), 'estimated_distance': est})
        return distances

# ────────────── Initialize ───────────────
model = YOLO(MODEL_DIR)
distance_calc = solutions.DistanceCalculation(model=model, show=False, line_width=2, show_conf=True, show_labels=True)
calc = InteractiveDistanceCalculator(distance_calc)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Insight AI system ready.", file=sys.stderr)

# ────────────── Main Loop ───────────────
while True:
    while not os.path.exists(TRIGGER_FILE):
        time.sleep(0.05)

    for _ in range(3):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        continue

    try:
        results, boxes = calc.process_frame(frame)
        h = frame.shape[0]
        nearby, all_objects = [], []

        if boxes:
            for b in boxes:
                d = estimate_distance_fallback(b, h)
                label = LABELS[int(b.cls[0])]
                all_objects.append((d, label))
                if d <= NEAR_THRESH_METRES:
                    nearby.append((d, label))

            # Debug output to stderr
            print(f"DEBUG: Detected {len(all_objects)} total objects", file=sys.stderr, flush=True)
            for dist, label in all_objects:
                print(f"DEBUG: {label} at {dist:.1f}m", file=sys.stderr, flush=True)
            print(f"DEBUG: {len(nearby)} objects within {NEAR_THRESH_METRES}m threshold", file=sys.stderr, flush=True)

            inter = calc.get_distances_between_objects(boxes)

            if not nearby:
                print("DEBUG: No nearby objects, removing trigger file", file=sys.stderr, flush=True)
                if os.path.exists(TRIGGER_FILE):
                    os.remove(TRIGGER_FILE)
                continue

            sentence = create_response_text(nearby)
            print(json.dumps({"text": sentence}, ensure_ascii=False), flush=True)
            print(sentence, file=sys.stderr, flush=True)

            response = {
                "text": sentence,
                "objects_detected": len(nearby),
                "details": [{"object": lbl, "distance_metres": round(float(dist), 1)} for dist, lbl in nearby],
                "inter_object_distances": [
                    {"object_pair": dist["objects"], "distance_metres": round(dist["estimated_distance"], 1)}
                    for dist in inter[:5]
                ]
            }

            # Atomic write using temp file
            temp_file = FEEDBACK_FILE + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(response, f)
                f.flush()
                os.fsync(f.fileno())
            os.rename(temp_file, FEEDBACK_FILE)

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)

    finally:
        if os.path.exists(TRIGGER_FILE):
            os.remove(TRIGGER_FILE)