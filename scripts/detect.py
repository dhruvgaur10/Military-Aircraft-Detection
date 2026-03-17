"""
Military Aircraft Detection - Detection Script
Author: Dhruv Gaur
GitHub: https://github.com/dhruvgaur10/Military-Aircraft-Detection
"""

from ultralytics import YOLO
import argparse
import os
import json
import subprocess
from collections import Counter
from datetime import datetime

# Aircraft reference data: country, role, and one key fact
AIRCRAFT_INFO = {
    "A10":        ("USA",        "Ground Attack",       "Built around the GAU-8 30mm rotary cannon"),
    "A400M":      ("Europe",     "Military Transport",  "NATO's primary tactical and strategic airlifter"),
    "AG600":      ("China",      "Amphibious",          "World's largest amphibious aircraft"),
    "B1":         ("USA",        "Strategic Bomber",    "Variable-sweep wing, supersonic at low altitude"),
    "B2":         ("USA",        "Stealth Bomber",      "Flying wing, radar cross-section of a small bird"),
    "B52":        ("USA",        "Strategic Bomber",    "In service since 1955, expected to fly past 2050"),
    "Be200":      ("Russia",     "Amphibious",          "Jet-powered, used for firefighting and SAR"),
    "C130":       ("USA",        "Tactical Transport",  "Most widely operated military transport globally"),
    "C17":        ("USA",        "Strategic Transport", "Can land on short, unimproved runways"),
    "C5":         ("USA",        "Heavy Transport",     "One of the largest aircraft ever built"),
    "E2":         ("USA",        "AEW&C",               "Carrier-based airborne early warning aircraft"),
    "EF2000":     ("Europe",     "Multirole Fighter",   "Eurofighter Typhoon, canard-delta for high agility"),
    "F117":       ("USA",        "Stealth Attack",      "First operational stealth aircraft, retired 2008"),
    "F14":        ("USA",        "Air Superiority",     "Variable-sweep Tomcat, retired from USN in 2006"),
    "F15":        ("USA",        "Air Superiority",     "Unbeaten in air combat with 100+ aerial victories"),
    "F16":        ("USA",        "Multirole Fighter",   "Most widely operated fighter aircraft in the world"),
    "F18":        ("USA",        "Multirole Fighter",   "Primary carrier-based strike fighter of the US Navy"),
    "F22":        ("USA",        "5th Gen Stealth",     "Supercruise capable, considered the top air superiority fighter"),
    "F35":        ("USA",        "5th Gen Multirole",   "Three variants covering USAF, Navy, and Marines"),
    "F4":         ("USA",        "Multirole Fighter",   "Cold War era Phantom II, over 5000 units built"),
    "J20":        ("China",      "5th Gen Stealth",     "China's first 5th generation stealth fighter"),
    "JAS39":      ("Sweden",     "Multirole Fighter",   "Gripen, known for very low cost per flight hour"),
    "MQ9":        ("USA",        "Armed UAV",           "Reaper, primary USAF hunter-killer drone"),
    "Mig31":      ("Russia",     "Interceptor",         "Mach 2.83 top speed, one of the fastest fighters"),
    "Mirage2000": ("France",     "Multirole Fighter",   "Delta wing design, operated by 9 countries"),
    "RQ4":        ("USA",        "Surveillance UAV",    "Global Hawk, high-altitude long-endurance drone"),
    "Rafale":     ("France",     "Multirole Fighter",   "Operates from land bases and aircraft carriers"),
    "SR71":       ("USA",        "Reconnaissance",      "Fastest air-breathing aircraft ever at Mach 3.3+"),
    "Su57":       ("Russia",     "5th Gen Stealth",     "Russia's first 5th generation stealth fighter"),
    "Tu160":      ("Russia",     "Strategic Bomber",    "Largest and heaviest combat aircraft ever built"),
    "Tu95":       ("Russia",     "Strategic Bomber",    "Turboprop bomber, still operational after 70 years"),
    "U2":         ("USA",        "Reconnaissance",      "Operates above 70,000 ft, used during Cold War"),
    "US2":        ("Japan",      "Amphibious",          "Search and rescue aircraft operated by JMSDF"),
    "V22":        ("USA",        "Tiltrotor",           "Osprey, takes off like a helicopter, flies like a plane"),
    "XB70":       ("USA",        "Experimental Bomber", "Valkyrie, Mach 3 prototype, only 2 were ever built"),
    "YF23":       ("USA",        "Prototype Fighter",   "Competed against YF-22, lost the ATF contract"),
}

# Confidence below this is flagged as uncertain
LOW_CONF_THRESHOLD = 0.50

# Aircraft that are visually similar and commonly confused by the model
SIMILAR_AIRCRAFT = {
    "Rafale":     ["JAS39", "Mirage2000"],
    "JAS39":      ["Rafale", "Mirage2000"],
    "Mirage2000": ["Rafale", "JAS39"],
    "F22":        ["YF23", "Su57", "J20"],
    "YF23":       ["F22", "Su57"],
    "Su57":       ["F22", "YF23", "J20"],
    "J20":        ["F22", "Su57"],
    "F15":        ["F16", "F18"],
    "F16":        ["F15", "F18"],
    "F18":        ["F15", "F16"],
    "B2":         ["F117"],
    "F117":       ["B2"],
    "B1":         ["Tu160"],
    "Tu160":      ["B1"],
    "MQ9":        ["RQ4"],
    "RQ4":        ["MQ9"],
    "Be200":      ["AG600", "US2"],
    "AG600":      ["Be200", "US2"],
    "US2":        ["Be200", "AG600"],
}


def play_result(path):
    """Open the result file with the system default application."""
    try:
        if os.name == 'nt':
            os.startfile(path)
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception:
        pass


def detect(source, model_path=None, confidence=0.25, play=False):
    """
    Run detection on an image, video, or webcam feed.

    Args:
        source    : Path to image/video or camera index (0 for webcam)
        model_path: Path to model weights file
        confidence: Minimum confidence threshold for detections
        play      : Open the result file after processing
    """
    print("=" * 55)
    print("  MILITARY AIRCRAFT DETECTION SYSTEM")
    print("=" * 55)

    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, 'models', 'best.pt')

    print(f"  Model      : {model_path}")
    print(f"  Source     : {source}")
    print(f"  Confidence : {confidence}")
    print("=" * 55)

    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=confidence,
        save=True,
        show=True
    )

    # Collect detections across all frames
    all_detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_score = float(box.conf[0])
            all_detections.append((cls_name, conf_score))

    print("=" * 55)
    print("  DETECTION SUMMARY")
    print("=" * 55)

    if not all_detections:
        print("  No aircraft detected in the source.")
    else:
        counts = Counter(name for name, _ in all_detections)
        best_conf = {}
        for name, conf_score in all_detections:
            if name not in best_conf or conf_score > best_conf[name]:
                best_conf[name] = conf_score

        print(f"  Total detections  : {len(all_detections)}")
        print(f"  Unique aircraft   : {len(counts)}")
        print()

        for name, count in counts.most_common():
            info = AIRCRAFT_INFO.get(name, ("Unknown", "Unknown", "No data available"))
            country, role, fact = info
            conf = best_conf[name]
            bar = "#" * int(conf * 20)
            print(f"  {name}")
            print(f"    Country : {country}  |  Role : {role}")
            print(f"    Fact    : {fact}")
            print(f"    Count   : x{count}   Confidence: {conf:.1%}  [{bar:<20}]")
            if conf < LOW_CONF_THRESHOLD:
                similar = SIMILAR_AIRCRAFT.get(name, [])
                print(f"    WARNING : Low confidence detection ({conf:.1%}). Model may be uncertain.")
                if similar:
                    print(f"    Could also be : {', '.join(similar)}")
            print()

        top_name = counts.most_common(1)[0][0]
        top_conf_name = max(best_conf, key=best_conf.get)
        print(f"  Most frequent      : {top_name}")
        print(f"  Highest confidence : {top_conf_name} ({best_conf[top_conf_name]:.1%})")

        # Save JSON report
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": str(source),
            "confidence_threshold": confidence,
            "total_detections": len(all_detections),
            "unique_aircraft": len(counts),
            "detections": [
                {
                    "aircraft": name,
                    "count": count,
                    "best_confidence": round(best_conf[name], 4),
                    "country": AIRCRAFT_INFO.get(name, ("Unknown",))[0],
                    "role": AIRCRAFT_INFO.get(name, (None, "Unknown",))[1],
                }
                for name, count in counts.most_common()
            ]
        }

        report_path = os.path.join(results[0].save_dir, "detection_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print()
        print(f"  Report saved to    : {report_path}")

    print(f"  Results saved to   : {results[0].save_dir}")
    print("=" * 55)

    if play and all_detections:
        save_dir = results[0].save_dir
        for fname in os.listdir(save_dir):
            if fname.endswith(('.mp4', '.avi', '.jpg', '.png')) and fname != 'detection_report.json':
                play_result(os.path.join(save_dir, fname))
                break

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Military Aircraft Detection')
    parser.add_argument('--source', type=str, required=True,
                        help='Image/video path or camera index (0 for webcam)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights (default: models/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--play', action='store_true',
                        help='Open the result file after processing')

    args = parser.parse_args()
    detect(args.source, args.model, args.conf, args.play)
