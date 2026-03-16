import numpy as np

class Track:

    def __init__(self, bbox, track_id):

        self.bbox = bbox
        self.id = track_id
        self.missing = 0


class SimpleTracker:

    def __init__(self, max_missing=10, iou_threshold=0.3):

        self.tracks = []
        self.next_id = 0
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold


    def iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB-xA) * max(0, yB-yA)

        boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

        union = boxAArea + boxBArea - interArea

        if union == 0:
            return 0

        return interArea / union


    def update(self, detections):

        updated_tracks = []

        for track in self.tracks:

            best_iou = 0
            best_det = None

            for det in detections:

                iou_score = self.iou(track.bbox, det)

                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det = det

            if best_iou > self.iou_threshold:

                track.bbox = best_det
                track.missing = 0

                updated_tracks.append(track)

                detections.remove(best_det)

            else:

                track.missing += 1

                if track.missing < self.max_missing:
                    updated_tracks.append(track)


        for det in detections:

            new_track = Track(det, self.next_id)

            self.next_id += 1

            updated_tracks.append(new_track)


        self.tracks = updated_tracks

        return [(t.id, t.bbox) for t in self.tracks]