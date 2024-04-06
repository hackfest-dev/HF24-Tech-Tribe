from gtts import gTTS
from playsound import playsound
from translate import Translator
from ultralytics import YOLO
import cv2
import tempfile
import os

# Function to speak text using gTTS
def speak(text, preferred_language_code):
    try:
        # Cleanse text input (example: remove special characters)
        cleaned_text = text.replace("$", "").replace(",", "")

        # Translate text to preferred language using the translate package
        translator = Translator(to_lang=preferred_language_code)
        translation = translator.translate(cleaned_text)

        # Use gTTS to convert translated text to speech
        tts = gTTS(text=translation, lang=preferred_language_code, slow=False)

        # Save the speech as a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            tts.save(temp_audio.name)
            temp_audio.close()

            # Play the audio file
            playsound(temp_audio.name)

        # Clean up the temporary audio file
        os.unlink(temp_audio.name)
    except Exception as e:
        print("Speech Error:", e)

# Load YOLO model
model = YOLO('best.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

 # Ask user for preferred language code
preferred_language_code = input("Enter the language code of your preferred language (e.g., 'hi' for Hindi, 'kn' for Kannada): ")

# Function to translate text to preferred language using the translate package
def translate_text(text, preferred_language_code):
    try:
        # Cleanse text input (example: remove special characters)
        cleaned_text = text.replace("$", "").replace(",", "")

        translator = Translator(to_lang=preferred_language_code)
        translation = translator.translate(cleaned_text)

        return translation
    except Exception as e:
        print("Translation Error:", e)
        return text  # Return the original text in case of error


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model(frame)

    # Extract class labels
    class_labels = [model.names[int(box.cls)] for box in results[0].boxes]

    # Convert class labels to text
    text = ", ".join(class_labels)

   

    # Translate text to preferred language using the translate package
    translated_text = translate_text(text, preferred_language_code)

    # Speak the translated text
    speak(translated_text, preferred_language_code)

    # Display annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
