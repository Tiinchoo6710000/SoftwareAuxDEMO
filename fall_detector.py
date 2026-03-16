import time
import numpy as np

class FallDetector:

    def __init__(self, min_time_on_ground=1.8):

        self.min_time = min_time_on_ground
        self.states = {}
        self.motion = {}

    def is_person_on_ground(self, bbox, keypoints):

        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1

        if h <= 0:
            return False

        aspect = w / h
        conditions = 0

        if aspect > 1.2:
            conditions += 1

        try:

            head = keypoints[0]
            sh_l = keypoints[5]
            sh_r = keypoints[6]
            hip_l = keypoints[11]
            hip_r = keypoints[12]

            if min(head[2], sh_l[2], sh_r[2], hip_l[2], hip_r[2]) < 0.25:
                return False

            avg_hip_y = (hip_l[1] + hip_r[1]) / 2
            avg_sh_y = (sh_l[1] + sh_r[1]) / 2

            if abs(head[1] - avg_hip_y) < h * 0.30:
                conditions += 1

            if abs(avg_sh_y - avg_hip_y) < h * 0.22:
                conditions += 1

        except:
            return False

        return conditions >= 2


    def update(self, pid, on_ground, bbox):

        now = time.time()

        if pid not in self.states:

            self.states[pid] = None

            self.motion[pid] = {
                "last_bbox": bbox,
                "still_since": now
            }

        last = self.motion[pid]["last_bbox"]

        if last is not None:

            dx = abs(bbox[0] - last[0])
            dy = abs(bbox[1] - last[1])

            # si se mueve resetea inmovilidad
            if dx + dy > 18:
                self.motion[pid]["still_since"] = now

        self.motion[pid]["last_bbox"] = bbox

        # lógica de caída por pose
        if on_ground:

            if self.states[pid] is None:
                self.states[pid] = now

        else:

            self.states[pid] = None


    def is_person_down(self, pid):

        if pid not in self.states or self.states[pid] is None:
            return False

        t = time.time()

        return (
            t - self.states[pid] >= self.min_time and
            t - self.motion[pid]["still_since"] >= 1.0
        )


    def is_person_still(self, pid, seconds=3.0):

        if pid not in self.motion:
            return False

        t = time.time()

        return (t - self.motion[pid]["still_since"]) >= seconds