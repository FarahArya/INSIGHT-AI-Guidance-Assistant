#!/usr/bin/env python3
"""
Complete Python-Only Vision Assistant for Visually Impaired
Enhanced with dual model support for objects + architectural elements
Optimized for Raspberry Pi 4
"""

import cv2
import json
import time
import numpy as np
import os
import sys
import threading
import queue
import shutil
from ultralytics import YOLO

# TTS Options - choose one based on your preference and system
import subprocess
import tempfile

class TTSEngine:
    """Text-to-Speech engine with speech state tracking"""

    def __init__(self, engine_type="piper"):
        self.engine_type = engine_type
        self.setup_engine()
        self.is_speaking = threading.Event()  # Track if TTS is active

    def setup_engine(self):
        if self.engine_type == "piper":
            # Using your existing Piper setup
            self.piper_cmd = [
                "./piper/piper",
                "--model", "./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx",
                "--config", "./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json",
                "--json-input"
            ]

        elif self.engine_type == "pyttsx3":
            import pyttsx3
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)

        elif self.engine_type == "gtts":
            import pygame
            pygame.mixer.init()

        elif self.engine_type == "espeak":
            pass

    def speak(self, text):
        """Speak the given text using the selected TTS engine"""
        self.is_speaking.set()  # Mark as speaking
        try:
            if self.engine_type == "piper":
                self._speak_piper(text)
            elif self.engine_type == "pyttsx3":
                self._speak_pyttsx3(text)
            elif self.engine_type == "gtts":
                self._speak_gtts(text)
            elif self.engine_type == "espeak":
                self._speak_espeak(text)
        except Exception as e:
            print(f"TTS Error: {e}", file=sys.stderr)
            try:
                self._speak_espeak(text)
            except:
                print(f"Speaking: {text}", file=sys.stderr)
        finally:
            self.is_speaking.clear()  # Mark as finished speaking

    def _speak_piper(self, text):
        """Use Piper TTS (your current setup)"""
        json_input = json.dumps({"text": text})

        subprocess.run(["amixer", "-q", "sset", "Headphone", "90%"], check=False)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_file = tmp_file.name

        player = "aplay"          # always use ALSA

        try:
            # Run Piper
            cmd = self.piper_cmd + ["--output_file", audio_file]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=json_input.encode())

            if process.returncode == 0:
                # Play audio file and wait for completion
                subprocess.run([player, audio_file], check=True)

            else:
                print(f"Piper error: {stderr.decode()}", file=sys.stderr)

        finally:
            if os.path.exists(audio_file):
                os.unlink(audio_file)

    def _speak_pyttsx3(self, text):
        """Use pyttsx3 (offline, cross-platform)"""
        self.engine.say(text)
        self.engine.runAndWait()

    def _speak_gtts(self, text):
        """Use Google TTS (requires internet)"""
        from gtts import gTTS
        import pygame

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            audio_file = tmp_file.name

        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(audio_file)

            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        finally:
            if os.path.exists(audio_file):
                os.unlink(audio_file)

    def _speak_espeak(self, text):
        """Use espeak (Linux, very lightweight)"""
        subprocess.run(["espeak", text], check=True)

class ImprovedDistanceEstimator:
    """Enhanced distance estimation with position detection"""

    def __init__(self):
        # Real-world object heights dictionary placeholder
        # ADD YOUR REAL_HEIGHTS DICTIONARY HERE
        # Format: {"Object_Name": height_in_meters, ...}
        self.REAL_HEIGHTS = {
            ## THIS IS A PLACE HOLDER
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
            "Table Tennis": 0.04,
            # Architectural elements (typical sizes)
            "door": 2.0,
            "window": 1.5,
            "wall": 3.0,  # Height reference for walls
        }

        # Real-world widths for dual estimation
self.REAL_WIDTHS = {
    # People and animals
    "Person": 0.50, "Dog": 0.40, "Cat": 0.25, "Horse": 0.60,
    "Cow": 0.60, "Sheep": 0.40, "Monkey": 0.30, "Bear": 0.90,
    "Elephant": 1.50, "Zebra": 0.50, "Giraffe": 0.80, "Deer": 0.50,
    "Duck": 0.20, "Pigeon": 0.20, "Wild Bird": 0.25, "Penguin": 0.30,
    "Parrot": 0.20, "Swan": 0.50, "Goose": 0.40, "Chicken": 0.25,
    "Pig": 0.40, "Camel": 0.70, "Antelope": 0.50, "Seal": 0.60,
    "Dolphin": 0.40, "Lion": 0.60, "Donkey": 0.40, "Yak": 0.70,
    "Rabbit": 0.20,

    # Vehicles
    "Car": 1.80, "SUV": 1.95, "Bus": 2.50, "Truck": 2.50,
    "Heavy Truck": 2.60, "Van": 2.00, "Motorcycle": 0.80, "Bicycle": 0.60,
    "Scooter": 0.70, "Train": 3.00, "Fire Truck": 2.50,
    "Ambulance": 2.30, "Sports Car": 1.85, "Pickup Truck": 2.00,
    "Formula 1": 1.90, "Tricycle": 0.60, "Trolley": 0.60,
    "Machinery Vehicle": 2.40,

    # Furniture
    "Chair": 0.45, "Table": 0.80, "Desk": 0.75, "Bed": 1.60,
    "Couch": 2.00, "Cabinet/shelf": 0.60, "Bench": 1.20,
    "Dining Table": 1.20, "Coffee Table": 0.90, "Nightstand": 0.45,
    "Side Table": 0.45, "Stool": 0.35,

    # Electronics
    "TV": 1.20, "Monitor/TV": 0.80, "Laptop": 0.35, "Computer Box": 0.35,
    "Printer": 0.40, "Speaker": 0.25, "Microwave": 0.45,
    "Refrigerator": 0.80, "Air Conditioner": 0.90, "Router/modem": 0.20,
    "Cell Phone": 0.07, "Telephone": 0.15, "Calculator": 0.10,

    # Sports equipment
    "Basketball": 0.24, "Soccer": 0.22, "Tennis": 0.06,
    "Baseball Bat": 0.07, "Golf Club": 0.10, "Hockey Stick": 0.08,
    "Tennis Racket": 0.30, "Skateboard": 0.20, "Surfboard": 0.60,
    "Snowboard": 0.30, "Skiboard": 0.15,

    # Containers
    "Backpack": 0.35, "Luggage": 0.45, "Handbag/Satchel": 0.35,
    "Storage box": 0.40, "Trash bin Can": 0.45, "Basket": 0.35,
    "Bowl/Basin": 0.25, "Barrel/bucket": 0.40,

    # Infrastructure
    "Traffic Light": 0.30, "Stop Sign": 0.75, "Fire Hydrant": 0.40,
    "Parking meter": 0.25, "Street Lights": 0.30, "Traffic Sign": 0.60,
    "Traffic cone": 0.30, "Crosswalk Sign": 0.60,

    # Architectural elements
    "door": 0.90, "window": 1.20, "wall": 0.30,
    "Sink": 0.60, "Toilet": 0.60, "Bathtub": 1.50,
    "Urinal": 0.35, "Showerhead": 0.15,

    # Appliances
    "Washing Machine/Drying Machine": 0.60, "Dishwasher": 0.60,
    "Oven": 0.60, "Gas stove": 0.60, "Rice Cooker": 0.30,
    "Coffee Machine": 0.25, "Toaster": 0.30, "Blender": 0.20,
    "Electric Drill": 0.10, "Hair Dryer": 0.15,

    # Default fallback
    "default": 0.30
}

        # Distance history for smoothing
        self.distance_history = {}

    def estimate_distance(self, box, img_h, img_w, label):
        """Enhanced distance estimation with better focal length and ground plane correction"""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        h_px = float(y2 - y1)
        w_px = float(x2 - x1)

        # Better focal length estimation based on image dimensions
        # Typical smartphone/webcam has ~70-degree horizontal FOV
        focal_px = img_w / (2 * np.tan(np.radians(35)))  # ~70 deg / 2

        real_h = self.REAL_HEIGHTS.get(label, None)

        if real_h:
            # Height-based estimation with better focal length
            distance_h = (real_h * focal_px) / h_px

            # Width-based estimation for validation (if we have width data)
            real_w = self.REAL_WIDTHS.get(label, None)

            if real_w:
                distance_w = (real_w * focal_px) / w_px
                # Average the two estimates, weighted by reliability
                distance = (distance_h * 0.7 + distance_w * 0.3)
            else:
                distance = distance_h

            # Ground plane correction for objects on ground
            if label in ["Person", "Car", "Bicycle", "Chair", "Dog", "Cat"]:
                # Assume camera is 1.5m high, pointing slightly down
                camera_height = 1.5
                box_bottom_y = float(y2)
                img_center_y = img_h / 2

                if box_bottom_y > img_center_y:  # Object below center (on ground)
                    # Simple ground plane correction
                    ground_factor = 1.0 + 0.3 * (box_bottom_y - img_center_y) / img_center_y
                    distance *= ground_factor

            # Special handling for walls - they can be very close or far
            if label == "wall":
                # For walls, use a different approach based on size
                wall_area = (h_px * w_px) / (img_h * img_w)
                if wall_area > 0.5:  # Wall takes up more than 50% of frame
                    distance = min(distance, 3.0)  # Cap very close walls

            return max(0.5, min(distance, 100.0))  # Clamp to reasonable range
        else:
            # Improved fallback for unknown objects
            apparent_size = (h_px * w_px) / (img_h * img_w)
            if apparent_size > 0.2:
                return 2.0
            elif apparent_size > 0.1:
                return 5.0
            elif apparent_size > 0.05:
                return 10.0
            else:
                return 20.0

    def get_object_position(self, box, img_width, left_threshold=0.33, right_threshold=0.67):
        """
        Determine if an object is positioned left, right, or forward (center) in the frame.

        Args:
            box: YOLO detection box object with xyxy coordinates
            img_width: Width of the image frame
            left_threshold: Fraction of image width defining left boundary (default: 0.33)
            right_threshold: Fraction of image width defining right boundary (default: 0.67)

        Returns:
            str: "left", "right", or "forward"
        """
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Calculate center x-coordinate of the object
        center_x = (x1 + x2) / 2

        # Normalize to 0-1 range
        normalized_x = center_x / img_width

        # Determine position
        if normalized_x < left_threshold:
            return "left"
        elif normalized_x > right_threshold:
            return "right"
        else:
            return "forward"

    def get_detailed_position(self, box, img_width, img_height):
        """
        Get more detailed position information including vertical position.

        Args:
            box: YOLO detection box object with xyxy coordinates
            img_width: Width of the image frame
            img_height: Height of the image frame

        Returns:
            dict: Dictionary with horizontal, vertical, and combined position info
        """
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Calculate center coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Normalize coordinates
        norm_x = center_x / img_width
        norm_y = center_y / img_height

        # Horizontal position
        if norm_x < 0.33:
            horizontal = "left"
        elif norm_x > 0.67:
            horizontal = "right"
        else:
            horizontal = "center"

        # Vertical position
        if norm_y < 0.33:
            vertical = "top"
        elif norm_y > 0.67:
            vertical = "bottom"
        else:
            vertical = "middle"

        # Combined description
        if horizontal == "center":
            if vertical == "middle":
                combined = "forward"
            else:
                combined = f"{vertical} forward"
        else:
            if vertical == "middle":
                combined = horizontal
            else:
                combined = f"{vertical} {horizontal}"

        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "combined": combined,
            "normalized_x": float(norm_x),
            "normalized_y": float(norm_y)
        }

    def _smooth_distance(self, label, distance, max_history=3):
        """Simple temporal smoothing to reduce jitter"""
        if label not in self.distance_history:
            self.distance_history[label] = []

        self.distance_history[label].append(distance)
        if len(self.distance_history[label]) > max_history:
            self.distance_history[label].pop(0)

        return sum(self.distance_history[label]) / len(self.distance_history[label])

class DualModelManager:
    """Manages dual YOLO models efficiently for Pi4"""

    def __init__(self, main_model_path, architectural_model_path):
        print("[INFO] Loading dual models...")

        # Load both models
        self.main_model = YOLO(main_model_path, task="detect")
        self.architectural_model = YOLO(architectural_model_path, task="detect")

        # Get model labels
        self.main_labels = self.main_model.names
        self.architectural_labels = self.architectural_model.names

        # Model alternation settings
        self.current_model = "main"
        self.model_switch_interval = 5  # Switch every 5 detections
        self.detection_count = 0

        # Performance optimization
        self.use_half_precision = True  # Use FP16 for faster inference on Pi4

        print(f"[INFO] Main model loaded with {len(self.main_labels)} classes")
        print(f"[INFO] Architectural model loaded with {len(self.architectural_labels)} classes")

    def get_current_model_info(self):
        """Get current active model and its labels"""
        if self.current_model == "main":
            return self.main_model, self.main_labels, "objects"
        else:
            return self.architectural_model, self.architectural_labels, "architecture"

    def should_switch_model(self):
        """Determine if it's time to switch models"""
        self.detection_count += 1
        if self.detection_count >= self.model_switch_interval:
            self.detection_count = 0
            self.current_model = "architectural" if self.current_model == "main" else "main"
            return True
        return False

    def run_detection(self, frame, conf_threshold=0.45):
        """Run detection with current model"""
        model, labels, model_type = self.get_current_model_info()

        # Optimize inference settings for Pi4
        results = model(
            frame,
            imgsz=480,  # Smaller image size for Pi4
            conf=conf_threshold,
            half=self.use_half_precision,  # Use FP16 if supported
            device='cpu',  # Explicitly use CPU
            verbose=False  # Reduce logging overhead
        )[0]

        return results, labels, model_type

    def get_combined_detection_summary(self, main_objects, arch_objects):
        """Combine and prioritize detections from both models"""
        combined = []

        # Add main objects
        for obj in main_objects:
            combined.append((*obj, "object"))

        # Add architectural elements with priority for navigation
        for obj in arch_objects:
            combined.append((*obj, "architecture"))

        # Sort by distance (closest first) but prioritize doors and walls
        def sort_key(item):
            distance, label, box, obj_type = item
            # Prioritize doors and walls that are close
            if label in ["door", "wall"] and distance < 3.0:
                return distance - 1.0  # Make them appear closer in sorting
            return distance

        combined.sort(key=sort_key)
        return combined

class VisionAssistant:
    """Main vision assistant class with dual model support"""

    def __init__(self, tts_engine="piper", main_model_path="./Insight/insight_deploy/models/yolo11n_object365.pt",
                 architectural_model_path="./models/yolo_architecture.pt"):
        print("[INFO] Initializing Vision Assistant with dual models...")

        # Initialize components
        self.tts = TTSEngine(tts_engine)
        self.distance_estimator = ImprovedDistanceEstimator()

        # Load dual models
        self.model_manager = DualModelManager(main_model_path, architectural_model_path)

        # Configuration
        self.conf_threshold = 0.45
        self.near_threshold = 6.0  # meters
        self.detection_interval = 1.5  # Increased interval for Pi4 performance
        self.speech_pause_time = 0.5

        # Separate thresholds for architectural elements
        self.arch_near_threshold = 4.0  # Closer threshold for walls/doors

        # Initialize camera with Pi4-optimized settings
        self.setup_camera()

        # Threading for TTS
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # Detection tracking
        self.last_detection_time = 0
        self.last_main_objects = []
        self.last_arch_objects = []

        print("[INFO] Dual-model Vision Assistant initialized successfully!")

    def setup_camera(self):
        """Initialize camera with Pi4-optimized settings"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        # Optimized settings for Pi4
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Reduced resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for Pi4
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer

        # Warm up camera
        for _ in range(3):
            self.cap.read()

        print("[INFO] Camera initialized with Pi4-optimized settings")

    def _tts_worker(self):
        """Background worker for TTS"""
        while True:
            try:
                text = self.tts_queue.get(timeout=1)
                if text is None:
                    break

                print(f"[TTS] Speaking: {text}")
                self.tts.speak(text)
                time.sleep(self.speech_pause_time)
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}", file=sys.stderr)

    def speak_async(self, text):
        """Add text to TTS queue"""
        try:
            while self.tts_queue.qsize() > 1:
                try:
                    old_text = self.tts_queue.get_nowait()
                    print(f"[TTS] Skipping: {old_text}")
                except queue.Empty:
                    break

            self.tts_queue.put(text)
        except Exception as e:
            print(f"Error queuing TTS: {e}", file=sys.stderr)

    def is_speaking(self):
        """Check if TTS is active"""
        return self.tts.is_speaking.is_set() or not self.tts_queue.empty()

    def objects_changed(self, new_objects, old_objects, threshold=2.0):
        """Check if objects have significantly changed"""
        if len(new_objects) != len(old_objects):
            return True

        new_sorted = sorted(new_objects, key=lambda x: (x[1], x[0]))
        old_sorted = sorted(old_objects, key=lambda x: (x[1], x[0]))

        for (new_dist, new_label, _), (old_dist, old_label, _) in zip(new_sorted, old_sorted):
            if new_label != old_label:
                return True
            if abs(new_dist - old_dist) > threshold:
                return True

        return False

    def create_dual_response_text(self, nearby_objects, arch_objects, frame_width):
        """Create response text combining both object types"""
        all_nearby = []

        # Add regular objects
        for dist, label, box in nearby_objects:
            position = self.distance_estimator.get_object_position(box, frame_width)
            all_nearby.append((dist, label, position, "object"))

        # Add architectural objects
        for dist, label, box in arch_objects:
            position = self.distance_estimator.get_object_position(box, frame_width)
            all_nearby.append((dist, label, position, "architecture"))

        if not all_nearby:
            return "No objects detected nearby."

        # Sort by distance and importance
        def sort_key(item):
            dist, label, pos, obj_type = item
            # Prioritize doors and walls
            if label in ["door", "wall"] and dist < 3.0:
                return dist - 1.0
            return dist

        all_nearby.sort(key=sort_key)

        # Limit to 3 most important objects
        all_nearby = all_nearby[:3]

        if len(all_nearby) == 1:
            dist, label, position, obj_type = all_nearby[0]
            return f"There is a {label} approximately {dist:.1f} metres {position}."

        parts = []
        for i, (dist, label, position, obj_type) in enumerate(all_nearby):
            if i == 0:
                parts.append(f"There is a {label} at {dist:.1f} metres {position}")
            elif i == len(all_nearby) - 1:
                parts.append(f"and a {label} at {dist:.1f} metres {position}")
            else:
                parts.append(f"a {label} at {dist:.1f} metres {position}")

        return ", ".join(parts) + "."

    def process_frame(self):
        """Process frame with dual model detection"""
        if self.is_speaking():
            print("[DEBUG] Skipping detection - TTS active")
            return None

        # Flush camera buffer
        for _ in range(2):
            self.cap.read()

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Run detection with current model
        results, labels, model_type = self.model_manager.run_detection(frame, self.conf_threshold)

        nearby_objects = []
        all_objects = []

        # Determine threshold based on model type
        threshold = self.arch_near_threshold if model_type == "architecture" else self.near_threshold

        for box in results.boxes:
            label = labels[int(box.cls[0])]
            distance = self.distance_estimator.estimate_distance(
                box, frame.shape[0], frame.shape[1], label
            )
            position = self.distance_estimator.get_object_position(box, frame.shape[1])

            all_objects.append((distance, label, position, model_type))

            if distance <= threshold:
                nearby_objects.append((distance, label, box))

        # Debug output
        print(f"[DEBUG] {model_type.upper()} model detected {len(all_objects)} objects", file=sys.stderr)
        for dist, label, position, obj_type in all_objects:
            print(f"[DEBUG] {obj_type}: {label} at {dist:.1f}m {position}", file=sys.stderr)

        # Check if we should switch models
        switched = self.model_manager.should_switch_model()
        if switched:
            print(f"[DEBUG] Switched to {self.model_manager.current_model} model", file=sys.stderr)

        return nearby_objects, frame.shape[1], frame.shape[0], model_type

    def run(self):
        """Main loop with dual model support"""
        print("[INFO] Starting Dual-Model Vision Assistant...")

        # Power-on announcement
        self.speak_async("Power on, Enhanced Insight with architectural detection is ready")

        try:
            while True:
                current_time = time.time()

                if (current_time - self.last_detection_time >= self.detection_interval and
                        not self.is_speaking()):

                    result = self.process_frame()

                    if result is not None:
                        nearby_objects, frame_width, frame_height, model_type = result

                        # Store results based on model type
                        if model_type == "architecture":
                            if self.objects_changed(nearby_objects, self.last_arch_objects):
                                # Announce architectural elements
                                if nearby_objects:
                                    response_text = self.create_dual_response_text([], nearby_objects, frame_width)
                                    if response_text and response_text != "No objects detected nearby.":
                                        print(f"[INFO] ARCHITECTURE: {response_text}")
                                        self.speak_async(response_text)
                                self.last_arch_objects = nearby_objects.copy()
                        else:
                            if self.objects_changed(nearby_objects, self.last_main_objects):
                                # Announce regular objects
                                if nearby_objects:
                                    response_text = self.create_dual_response_text(nearby_objects, [], frame_width)
                                    if response_text and response_text != "No objects detected nearby.":
                                        print(f"[INFO] OBJECTS: {response_text}")
                                        self.speak_async(response_text)
                                self.last_main_objects = nearby_objects.copy()

                        self.last_detection_time = current_time

                # Reduced sleep for better responsiveness
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[INFO] Shutting down Vision Assistant...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        self.tts_queue.put(None)
        self.tts_thread.join(timeout=3)
        print("[INFO] Vision Assistant stopped")

def main():
    """Main function with dual model configuration"""

    # Configuration - adjust paths based on your setup
    TTS_ENGINE = "piper"
    MAIN_MODEL_PATH = "./Insight/insight_deploy/models/yolo11n_object365.pt"
    ARCHITECTURAL_MODEL_PATH = "./Insight/insight_deploy/models/best3.pt"  # Path to your architecture model

    # Check if architectural model exists
    if not os.path.exists(ARCHITECTURAL_MODEL_PATH):
        print(f"[WARNING] Architectural model not found at {ARCHITECTURAL_MODEL_PATH}")
        print("[INFO] Please ensure you have a trained model for walls, doors, and windows")
        print("[INFO] Falling back to single model mode...")
        # You could fallback to single model here if needed
        sys.exit(1)

    try:
        assistant = VisionAssistant(
            tts_engine=TTS_ENGINE,
            main_model_path=MAIN_MODEL_PATH,
            architectural_model_path=ARCHITECTURAL_MODEL_PATH
        )
        assistant.run()
    except Exception as e:
        print(f"[ERROR] Failed to start Vision Assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()