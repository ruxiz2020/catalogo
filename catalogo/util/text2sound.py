import os
from gtts import gTTS

#from os import environ
#environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
#from pygame import mixer
#from time import sleep


class Test2Sound:

    def __init__(self, text, language, outfile="../test/data/test.mp3"):
        """
        Extract convert text to sound
        """
        self._text = text
        self._language = language
        self._outfile = outfile

        self.text2sound()
        self.save_sound_2_mp3()

    def text2sound(self):

        self.sound_object = gTTS(text=self._text,
                            lang=self._language,
                            slow=False)  # speed

    def save_sound_2_mp3(self):

        self.sound_object.save(self._outfile)


    #def read_sound_file(self):

    #    mixer.init()
    #    mixer.music.load(self._outfile)
    #    mixer.music.play(1)
    #    while mixer.music.get_busy():
    #        sleep(1)
    #    print("Finished playing sound file")
        #playsound.playsound(sound_file, False)


if __name__ == '__main__':

    data = "Insanity: doing the same thing over and over again and expecting different results."

    ts = Test2Sound(text=data, language='en')

    
