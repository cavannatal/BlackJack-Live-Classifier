import cv2
from ultralytics import YOLO
import torch
import os
import numpy as np


def detect_cards_and_get_values(results):
    cards = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = box.conf.cpu().item()
            bbox = box.xyxy[0].cpu().numpy()
            cards.append((class_id, confidence, bbox))
    return cards


def get_card_name(class_id):

    card_names = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S',
                  '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S',
                  '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S',
                  'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS',
                  'QC', 'QD', 'QH', 'QS']
    return card_names[class_id]


def calculate_hand_value(cards):
    value = 0
    aces = 0
    for card in cards:
        card_name = get_card_name(card[0])
        rank = card_name[:-1]
        if rank in ['J', 'Q', 'K']:
            value += 10
        elif rank == 'A':
            aces += 1
            value += 11
        else:
            value += int(rank)

    while value > 21 and aces:
        value -= 10
        aces -= 1

    return value


def combine_close_detections(detections, threshold=50):
    combined_detections = []
    used = [False] * len(detections)

    for i in range(len(detections)):
        if used[i]:
            continue

        class_id, confidence, bbox = detections[i]
        x1, y1, x2, y2 = bbox
        close_boxes = [(class_id, confidence, bbox)]
        used[i] = True

        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            class_id2, confidence2, bbox2 = detections[j]
            x1_2, y1_2, x2_2, y2_2 = bbox2


            center_x1 = (x1 + x2) / 2
            center_y1 = (y1 + y2) / 2
            center_x2 = (x1_2 + x2_2) / 2
            center_y2 = (y1_2 + y2_2) / 2
            distance = np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)

            if distance < threshold:
                close_boxes.append((class_id2, confidence2, bbox2))
                used[j] = True


        if close_boxes:
            avg_class_id = np.bincount([box[0] for box in close_boxes]).argmax()
            avg_confidence = np.mean([box[1] for box in close_boxes])
            avg_bbox = np.mean([box[2] for box in close_boxes], axis=0)
            combined_detections.append((avg_class_id, avg_confidence, avg_bbox))

    return combined_detections


def main():

    deck = {i: 1 for i in range(52)}


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    model_path = 'C:/Users/CAVAN/runs/detect/train11/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break


        height, width, _ = frame.shape


        top_half = frame[:height // 2, :]
        bottom_half = frame[height // 2:, :]


        results_top = model(top_half, device=device)
        top_cards = detect_cards_and_get_values(results_top)


        results_bottom = model(bottom_half, device=device)
        bottom_cards = detect_cards_and_get_values(results_bottom)


        top_cards_combined = combine_close_detections(top_cards)
        bottom_cards_combined = combine_close_detections(bottom_cards)


        top_cards_combined = [card for card in top_cards_combined if card[1] > 0.9 and deck[card[0]] > 0]
        bottom_cards_combined = [card for card in bottom_cards_combined if card[1] > 0.9 and deck[card[0]] > 0]

        for card in top_cards_combined:
            if deck[card[0]] > 0:
                deck[card[0]] -= 1

        for card in bottom_cards_combined:
            if deck[card[0]] > 0:
                deck[card[0]] -= 1


        annotated_top = results_top[0].plot()
        annotated_bottom = results_bottom[0].plot()
        annotated_frame = cv2.vconcat([annotated_top, annotated_bottom])


        dealer_value = calculate_hand_value(top_cards_combined)
        player_value = calculate_hand_value(bottom_cards_combined)
        decision = "Hit!" if player_value <= 16 else "Stay!"

        cv2.putText(annotated_frame, f"Dealer's Value: {dealer_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Player's Value: {player_value}", (10, height // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Decision: {decision}", (10, height // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Webcam - YOLOv8 Card Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
