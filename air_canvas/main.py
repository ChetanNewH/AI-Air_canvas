import cv2
import mediapipe as mp
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button


class AirCanvasApp(App):

    def build(self):

        self.capture = None

        self.prev_x = 0
        self.prev_y = 0

        self.brush_thickness = 7
        self.eraser_thickness = 30

        self.draw_color = (255,0,0)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)

        self.mp_draw = mp.solutions.drawing_utils

        layout = BoxLayout(orientation="vertical")

        self.image = Image()
        layout.add_widget(self.image)

        buttons = BoxLayout(size_hint_y=0.2)

        start_btn = Button(text="Start")
        stop_btn = Button(text="Stop")
        clear_btn = Button(text="Clear")

        start_btn.bind(on_press=self.start_camera)
        stop_btn.bind(on_press=self.stop_camera)
        clear_btn.bind(on_press=self.clear_canvas)

        buttons.add_widget(start_btn)
        buttons.add_widget(stop_btn)
        buttons.add_widget(clear_btn)

        layout.add_widget(buttons)

        return layout


    def start_camera(self,instance):

        self.capture = cv2.VideoCapture(0)

        ret,frame = self.capture.read()

        if ret:
            h,w,_ = frame.shape
            self.canvas_layer = np.zeros((h,w,3),dtype=np.uint8)

        Clock.schedule_interval(self.update,1/30)


    def stop_camera(self,instance):

        Clock.unschedule(self.update)

        if self.capture:
            self.capture.release()


    def clear_canvas(self,instance):

        self.canvas_layer[:] = 0


    def fingers_up(self,hand):

        fingers=[]

        if hand.landmark[8].y < hand.landmark[6].y:
            fingers.append(1)
        else:
            fingers.append(0)

        if hand.landmark[12].y < hand.landmark[10].y:
            fingers.append(1)
        else:
            fingers.append(0)

        return fingers


    def update(self,dt):

        ret,frame = self.capture.read()

        if not ret:
            return

        frame = cv2.flip(frame,1)

        # COLOR BUTTONS
        cv2.rectangle(frame,(10,10),(110,60),(255,0,0),-1)
        cv2.rectangle(frame,(120,10),(220,60),(0,255,0),-1)
        cv2.rectangle(frame,(230,10),(330,60),(0,0,255),-1)
        cv2.rectangle(frame,(340,10),(440,60),(0,0,0),-1)

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:

            for hand in result.multi_hand_landmarks:

                h,w,_ = frame.shape

                x = int(hand.landmark[8].x*w)
                y = int(hand.landmark[8].y*h)

                fingers = self.fingers_up(hand)

                # COLOR SELECTION
                if y < 60:

                    if 10 < x < 110:
                        self.draw_color = (255,0,0)

                    elif 120 < x < 220:
                        self.draw_color = (0,255,0)

                    elif 230 < x < 330:
                        self.draw_color = (0,0,255)

                    elif 340 < x < 440:
                        self.draw_color = (0,0,0)

                # DRAW MODE
                if fingers == [1,0]:

                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x,self.prev_y = x,y

                    cv2.line(self.canvas_layer,
                             (self.prev_x,self.prev_y),
                             (x,y),
                             self.draw_color,
                             self.brush_thickness)

                    self.prev_x,self.prev_y = x,y

                # ERASER MODE
                elif fingers == [1,1]:

                    cv2.circle(self.canvas_layer,(x,y),
                               self.eraser_thickness,
                               (0,0,0),
                               -1)

                    self.prev_x,self.prev_y = 0,0

                self.mp_draw.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS)

        frame = cv2.add(frame,self.canvas_layer)

        buf = cv2.flip(frame,0).tobytes()

        texture = Texture.create(
            size=(frame.shape[1],frame.shape[0]),
            colorfmt='bgr')

        texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')

        self.image.texture = texture


if __name__ == "__main__":
    AirCanvasApp().run()