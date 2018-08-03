import pyaudio
import wave
from array import array
from scipy.io import wavfile
from scipy import signal
from scipy import misc
import numpy as np
import math
from scipy.signal import lfilter
from scikits.talkbox import lpc
import time
import matplotlib.pyplot as plt
from divapy import Diva as Divapy
import os
from sklearn import preprocessing
import pandas as pd
import skimage.measure

max_F0 = 200.

def record_sound(file_name, time, rate=11025, FORMAT=pyaudio.paInt16, CHANNELS=1, chunk=1024):
    #Press  key for start recording
    raw_input("Press Enter to start recording...")

    audio=pyaudio.PyAudio() #instantiate the pyaudio

    #recording prerequisites
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=rate, input=True, frames_per_buffer=chunk)

    #starting recording
    frames=[]

    for i in range(0, int(rate / chunk * time)):
        data=stream.read(chunk)
        data_chunk=array('h',data)
        vol=max(data_chunk)
        if(vol>=500):
            print("something said")
            frames.append(data)
        else:
            print("nothing")
        print("\n")

    #end of recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    #writing to file
    wavfile_ = wave.open(file_name, 'wb')
    wavfile_.setnchannels(CHANNELS)
    wavfile_.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile_.setframerate(rate)
    wavfile_.writeframes(b''.join(frames))#append frames recorded to file
    wavfile_.close()


def load_sound(file_name):
    fs, sound = wavfile.read(file_name)
    return sound

def play_sound(sound):  # keep in mind that DivaMatlab works with ts=0.005
    pa = pyaudio.PyAudio()  # If pa and stream are not elements of the self object then sound does not play
    stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                          rate=11025,
                          output=True)
    stream.start_stream()
    stream.write(sound.astype(np.int16).tostring())
    time.sleep(len(sound)/11025. + 0.2)

def get_fromants(sound, fs=11025):
    # Read from file.
    # spf = sound

    x = sound
    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1.], [1., 0.63], x1)

    # Get LPC.
    ncoeff = 2 + fs / 1000
    A, e, k = lpc(x1, ncoeff)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Fs = spf.getframerate() #Gregory comment
    frqs = sorted(angz * (fs / (2 * math.pi)))

    return frqs

def plot_formants(formants, labels = None):
    i = 0
    for formant in formants:
        #print(formant)
        plt.plot(formant[1], formant[2], 'ob')
        plt.annotate(labels[i], xy=(formant[1], formant[2]))
        i += 1

def record_vt_sound(art,time, fs = 11025):
    diva_synth = Divapy()
    n_arts = round((time/0.005))+1
    arts = np.tile(art, (int(n_arts), 1))
    sound = diva_synth.get_sound(arts)
    scaled = np.int16(sound / np.max(np.abs(sound)) * 32767)
    wavfile.write("vt" + '.wav', 11025, scaled)
    diva_synth.play_sound(sound)


def create_german_files(german_art, time, fs=11025):
    diva_synth = Divapy()
    n_arts = round((time/0.005))+1
    vow_names = ['-E', '-E_', '-I', '-O', '-U', '-Y', '-Z_', '-a', '-a_', '-at', '-b', '-e_', '-i_', '-o_', '-p', '-u_', '-y']
    for i in range(len(german_art[:, 0])):
        german_vowels = german_art[i, :]
        german_vowels = np.tile(german_vowels, (int(n_arts), 1))
        german_sound = diva_synth.get_sound(german_vowels)
        scaled = np.int16(german_sound / np.max(np.abs(german_sound)) * 32767)
        wavfile.write("vt" + vow_names[i] + '.wav', fs, scaled)
    #diva_synth.play_sound(german_sound)

#### ---> INT TO FLOAT AND FLOAT TO INT <--- ####
def int_to_float(data):
    float_data = map(float, data)
    # float_data = np.memmap(data, dtype='float32')
    return float_data

def float_to_int(data):
    int_data = map(int, data)
    # int_data = np.memmap(data, dtype='int16')
    return int_data

def record_sound_save_to_folder(file_name, time, directory, rate=11025, FORMAT=pyaudio.paInt16, CHANNELS=1, chunk=1024):
    #Press  key for start recording
    raw_input("Press Enter to start recording...")
    audio=pyaudio.PyAudio() #instantiate the pyaudio
    #recording prerequisites
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=rate, input=True, frames_per_buffer=chunk)
    #starting recording
    frames=[]
    for i in range(0, int(rate / chunk * time)):
        data=stream.read(chunk)
        data_chunk=array('h',data)
        vol=max(data_chunk)
        if(vol>=500):
            print("something said")
            frames.append(data)
        else:
            print("nothing")
        print("\n")

    #end of recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    #writing to file
    directory = directory
    completeName = os.path.join(directory, file_name)
    wavfile_ = wave.open(completeName, 'wb')
    wavfile_.setnchannels(CHANNELS)
    wavfile_.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile_.setframerate(rate)
    wavfile_.writeframes(b''.join(frames))#append frames recorded to file
    wavfile_.close()

def creat_base_of_files(list_of_words, t):
    person = raw_input('Enter your name: ')
    print('Hello ' + person)
    #creating folder with name of the person
    directory = './'+person+'/'
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)
    files = []
    #creating list of name (name_word_1_iteration_2.wav)
    for i in list_of_words:
        files.append(person + "_word_" + i + "_iteration_")
    #record sound and save in file
    for i, file__ in enumerate(files):
        for j in range(1, 3):
            file_ = file__ + str(j) + ".wav"
            print("Next sound " + list_of_words[i])
            record_sound_save_to_folder(file_ , t, directory)
            print("Created sound " + str(file_))

            completeName = os.path.join(directory, file_)
            print(completeName)
            sound = load_sound(completeName)
            play_sound(sound)

            answer = raw_input("Sound its correct? Y/N\n")
            while answer != 'y':
                os.remove(completeName)
                print("Repeating sound " + str(file_))
                record_sound_save_to_folder(file_, t, directory)
                sound = load_sound(completeName)
                play_sound(sound)
                print("Created sound " + str(file_))
                answer = raw_input("Sound its correct? Y/N\n")
    return files, directory

def get_chunks(sound, t_sw=50, t_ol=25, fs = 11025): #time in miliseconds
    width_size = int(math.ceil((t_sw/1000.)*fs) + 1)
    print (width_size)
    width_overlaping = int(math.ceil((t_ol/1000.)*fs) + 1)
    size_of_sound = len(sound)
    step_move = width_size - width_overlaping
    list_of_chunks=[]
    for j in range(0, size_of_sound - (width_size - width_overlaping), step_move):
        sub_samples = sound[j:width_size]
        width_size = width_size + step_move
        list_of_chunks.append([sub_samples])
    return list_of_chunks

def get_formants_trayectory(list_of_chunks):
    chunks_formants = [get_fromants(chunk[0]) for chunk in list_of_chunks]
    for i in range(len(chunks_formants)):
        created_formants_array = np.array(chunks_formants[i])
        non_zero_index = np.nonzero(created_formants_array)[0][0]
        # print (non_zero_index)
        if created_formants_array[non_zero_index] > max_F0:
            chunks_formants[i] = [0.] + list(created_formants_array[non_zero_index:non_zero_index+4])
        else:
            chunks_formants[i] = list(created_formants_array[non_zero_index:non_zero_index+5])
    return chunks_formants


def get_scaling(chunks_formants):
    x = preprocessing.MinMaxScaler().fit_transform(chunks_formants)
    normalization_data = pd.DataFrame(x)
    return normalization_data

def compare_kernels(image, *args):
    for arg in args:
        temp_image = misc.np.array(image).T
        Conv = signal.convolve2d(temp_image, arg)
        ReLu = np.maximum(Conv, 0)
        reduced_image = skimage.measure.block_reduce(ReLu, (2, 2), np.max)
        plt.figure()
        plt.imshow(reduced_image.T, cmap='gray')

    plt.show()
    return reduced_image


def apply_conv(image, *args):
    temp_image = misc.np.array(image).T
    plt.subplot(len(args)+1, 1, 1)
    plt.imshow(image, cmap='gray')
    for i, arg in enumerate(args):
        Conv = signal.convolve2d(temp_image, arg)
        ReLu = np.maximum(Conv, 0)
        temp_image = skimage.measure.block_reduce(ReLu, (2, 2), np.max)

        plt.subplot(len(args)+1, 1, i + 2)
        plt.imshow(temp_image.T, cmap='gray')

    plt.show()
    return temp_image

if __name__ == '__main__':
    full_name_file =  directory = './' + 'marcin' + '/' + 'marcin_word_bat_iteration_1.wav'
    loaded_files = load_sound(full_name_file)

    chunks_list = get_chunks(loaded_files)
    chunk_formants = get_formants_trayectory(chunks_list)

    normalization = get_scaling(chunk_formants)

    edge_detection = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    sharpen_image = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    list_krenels2 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    # reduced_image = compare_kernels(np.array(normalization).T, edge_detection, sharpen_image, list_krenels2)
    new_image = apply_conv(np.array(normalization).T, edge_detection, sharpen_image, list_krenels2)
    # plt.subplot(2, 1, 1)
    # plt.imshow(reduced_image.T, cmap='gray')
    # plt.subplot(2, 1, 2)
    # plt.imshow(np.array(normalization).T, cmap='gray')
    #
    # plt.title(full_name_file)
    # plt.show()


# words = ['hello', 'who', 'which', 'where', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'one', 'two', 'three', 'four',
#         'five', 'six', 'seven', 'eight', 'nine', 'ten', 'red', 'blue', 'green', 'white', 'black', 'orange', 'pink',
#         'yellow', 'hours', 'duck', 'pig', 'cow', 'chicken', 'dog', 'cat', 'mouse', 'rat', 'bee', 'bat', 'lion','house'
#         'block', 'flat', 'line', 'square', 'triangle', 'circle', 'cube']