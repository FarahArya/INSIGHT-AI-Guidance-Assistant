#!/usr/bin/env python3
import cv2, json, time, numpy as np, os
from ultralytics import YOLO, solutions
import sys

"""
Insight – YOLOv11n (NCNN) → Piper TTS
Run on Raspberry Pi 4:
 python3 insight_infer.py | \
 piper --model /home/pi/voices/en_GB-alba-low.onnx --json-input
"""
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
# ───────────────────────── CONFIG ──────────────────────────
MODEL_DIR = os.getenv('MODEL_PATH', './models/yolo11n_object365.pt')
LABELS = YOLO(MODEL_DIR).names
FOCAL_PX = 600
CONF_THRES = 0.45
NEAR_THRESH_METRES = 6  # Increased from 5 to 6 meters
TRIGGER_FILE = os.getenv('TRIGGER_PATH', '../../shared/trigger.txt')
FEEDBACK_FILE = os.getenv('FEEDBACK_PATH', '../../shared/feedback.json')

# ─────────────────────────────────────────────────────────────
# Load YOLO model and distance calculator class
# Initialize distance calculation object using Ultralytics solutions
distancecalculator = solutions.DistanceCalculation(
    model=MODEL_DIR,
    show=False,  # set to True if you want to display the output
    line_width=2,
    show_conf=True,
    show_labels=True
)
model = YOLO(MODEL_DIR, task="detect")
# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def estimate_distance_fallback(box, img_h):
    """Fallback distance estimation using the original method"""
    x1, y1, x2, y2 = box.xyxy[0]
    h_px = float(y2 - y1)
    label = LABELS[int(box.cls[0])]
    real_h = REAL_HEIGHTS.get(label, None)
    if real_h:
        return (real_h * FOCAL_PX) / h_px
    return (img_h / h_px) * 0.5


def get_box_centroid(box):
    """Get centroid of bounding box"""
    x1, y1, x2, y2 = box.xyxy[0]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_euclidean_distance(point1, point2, pixels_per_meter=None):
    """Calculate Euclidean distance between two points"""
    if pixels_per_meter is None:
        # Use a reasonable default or calculate based on your setup
        pixels_per_meter = 100  # Adjust based on your camera setup

    pixel_distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return pixel_distance / pixels_per_meter


def create_response_text(nearby_objects):
    """Create natural language response for all nearby objects"""
    if len(nearby_objects) == 1:
        dist, label = nearby_objects[0]
        return f"There is a {label} approximately {dist:.0f} metres ahead."

    # Sort by distance for better readability
    nearby_objects.sort(key=lambda x: x[0])

    if len(nearby_objects) == 2:
        obj1, obj2 = nearby_objects
        return f"There is a {obj1[1]} approximately {obj1[0]:.0f} metres ahead, and a {obj2[1]} at {obj2[0]:.0f} metres."

    # For 3+ objects
    parts = []
    for i, (dist, label) in enumerate(nearby_objects):
        if i == 0:
            parts.append(f"There is a {label} at {dist:.0f} metres")
        elif i == len(nearby_objects) - 1:
            parts.append(f"and a {label} at {dist:.0f} metres ahead")
        else:
            parts.append(f"a {label} at {dist:.0f} metres")

    return ", ".join(parts) + "."


class InteractiveDistanceCalculator:
    """Enhanced distance calculator that combines Ultralytics solutions with custom logic"""

    def __init__(self, model_path, show=False):
        self.distance_calc = solutions.DistanceCalculation(
            model=model_path,
            show=show,
            line_width=2,
            show_conf=True,
            show_labels=True
        )
        self.selected_objects = []
        self.frame_count = 0

    def process_frame(self, frame):
        """Process frame with distance calculation"""
        # Use Ultralytics distance calculation
        results = self.distance_calc(frame)

        # Get detection results
        if hasattr(results, 'boxes') and results.boxes is not None:
            return results, results.boxes
        else:
            # Fallback to regular YOLO detection
            detection_results = model(frame, imgsz=640, conf=CONF_THRES)[0]
            return results, detection_results.boxes

    def get_distances_between_objects(self, boxes):
        """Calculate distances between all detected objects"""
        distances = []
        centroids = []

        # Get centroids of all detected objects
        for box in boxes:
            centroid = get_box_centroid(box)
            label = LABELS[int(box.cls[0])]
            centroids.append((centroid, label, box))

        # Calculate distances between all pairs
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                point1, label1, box1 = centroids[i]
                point2, label2, box2 = centroids[j]

                # Calculate Euclidean distance in pixels
                pixel_distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

                # Convert to approximate real-world distance
                # This is a simplified conversion - you may need to calibrate this
                estimated_distance = pixel_distance / 100  # Adjust this factor based on your setup

                distances.append({
                    'objects': (label1, label2),
                    'pixel_distance': pixel_distance,
                    'estimated_distance': estimated_distance,
                    'centroids': (point1, point2)
                })

        return distances


enhanced_calc = InteractiveDistanceCalculator(MODEL_DIR, show=False)

print("Enhanced Distance Calculation System Started", file=sys.stderr, flush=True)
print("Using Ultralytics YOLO11 Distance Calculation Solutions", file=sys.stderr, flush=True)

while True:
    # Wait for trigger
    while not os.path.exists(TRIGGER_FILE):
        time.sleep(0.05)

    # Flush camera buffer to get fresh frame
    for _ in range(3):
        cap.read()

    ok, frame = cap.read()
    if not ok:
        continue

    try:
        # Process frame with enhanced distance calculation
        results, boxes = enhanced_calc.process_frame(frame)
        h = frame.shape[0]

        # Get all objects with their distances using your original method
        nearby_objects = []
        all_objects = []

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                d = estimate_distance_fallback(b, h)
                label = LABELS[int(b.cls[0])]
                all_objects.append((d, label))

                if d <= NEAR_THRESH_METRES:
                    nearby_objects.append((d, label))

            # Also calculate inter-object distances using Ultralytics approach
            inter_distances = enhanced_calc.get_distances_between_objects(boxes)

            # Debug output
            print(f"DEBUG: Detected {len(all_objects)} total objects", file=sys.stderr, flush=True)
            for dist, label in all_objects:
                print(f"DEBUG: {label} at {dist:.1f}m", file=sys.stderr, flush=True)

            # Log inter-object distances
            if inter_distances:
                print(f"DEBUG: Inter-object distances:", file=sys.stderr, flush=True)
                for dist_info in inter_distances[:3]:  # Show first 3 pairs
                    obj1, obj2 = dist_info['objects']
                    est_dist = dist_info['estimated_distance']
                    print(f"DEBUG: {obj1} to {obj2}: {est_dist:.1f}m", file=sys.stderr, flush=True)

            print(f"DEBUG: {len(nearby_objects)} objects within {NEAR_THRESH_METRES}m threshold", file=sys.stderr,
                  flush=True)

        if not nearby_objects:
            print("DEBUG: No nearby objects, removing trigger file", file=sys.stderr, flush=True)
            os.remove(TRIGGER_FILE)
            continue

        # Create response for all nearby objects
        sentence = create_response_text(nearby_objects)

        # Send to stdout for piping to TTS
        print(json.dumps({"text": sentence}, ensure_ascii=False), flush=True)
        print(sentence, file=sys.stderr, flush=True)

        # Write detailed feedback to file
        try:
            response = {
                "text": sentence,
                "objects_detected": len(nearby_objects),
                "details": [{"object": label, "distance_metres": round(float(dist), 1)} for dist, label in
                            nearby_objects],
                "inter_object_distances": [
                    {
                        "object_pair": dist_info['objects'],
                        "distance_metres": round(float(dist_info['estimated_distance']), 1)
                    }
                    for dist_info in inter_distances[:5]  # Include first 5 pairs
                ] if 'inter_distances' in locals() else []
            }

            # Atomic write
            temp_file = FEEDBACK_FILE + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(response, f)
                f.flush()
                os.fsync(f.fileno())

            os.rename(temp_file, FEEDBACK_FILE)

        except Exception as e:
            print(f"Error writing to feedback file: {e}", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"Error processing frame: {e}", file=sys.stderr, flush=True)

    finally:
        # Remove trigger file to indicate completion
        if os.path.exists(TRIGGER_FILE):
            os.remove(TRIGGER_FILE)
