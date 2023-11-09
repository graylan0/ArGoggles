import cv2
import mediapipe as mp
import threading
import base64
import requests
import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

openai.api_key = 'your-api-key'

class HandDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.lock = threading.Lock()
        self.frame = None
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            success, img = self.cap.read()
            if success:
                self.lock.acquire()
                self.frame = img
                self.lock.release()

    def get_frame(self):
        self.lock.acquire()
        frame = self.frame
        self.lock.release()
        return frame

    def find_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def close(self):
        self.cap.release()

class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        frame = self.capture.get_frame()
        if frame is not None:
            frame = self.capture.find_hands(frame)
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture

class GameApp(App):
    def build(self):
        self.detector = HandDetector()
        self.img1 = KivyCamera(capture=self.detector, fps=30)
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        self.send_button = Button(text='Pop Bubble')
        self.send_button.bind(on_press=self.on_enter)
        layout.add_widget(self.send_button)
        return layout

    def on_enter(self, instance):
        frame = self.detector.get_frame()
        encoded_image = self.encode_image_to_base64(frame)
        self.send_image_to_gpt4(encoded_image)

    def encode_image_to_base64(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def send_image_to_gpt4(self, base64_image):
        headers = {
            'Authorization': f'Bearer {openai.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'gpt-4-vision-preview',
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'What’s in this image?'},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}
                    ]
                }
            ],
            'max_tokens': 300
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        self.interpret_response(response.json())

    def interpret_response(self, response):
        # This is a placeholder for interpreting the response from GPT-4 Vision.
        # You would extract the relevant information from the response and use it to pop a bubble.
        try:
            text_response = response['choices'][0]['message']['content']
            print(text_response)
            # Here you would have logic to interpret the response and decide which bubble to pop.
            # For example:
            # if "pop the red bubble" in text_response:
            #     self.pop_bubble('red')
        except KeyError:
            print("Error in response:", response)

    def pop_bubble(self, color):
        # Placeholder function for game logic to pop a bubble.
        print(f"Popping {color} bubble!")

    def on_stop(self):
        self.detector.close()

if __name__ == '__main__':
    GameApp().run()
