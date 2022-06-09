import argparse
import json
import numpy as np
import os
import sys
sys.path.append('/home/ivan/catkin_ws/src/lebron_bot_2/scripts')
sys.path.append('/home/ivan/catkin_ws/src/lebron_bot_2/data')
sys.path.append('/home/ivan/catkin_ws/src/lebron_bot_2/model')
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


messageB = ""
class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '/home/ivan/catkin_ws/src/lebron_bot_2/model/charList.txt'
    fn_summary = '/home/ivan/catkin_ws/src/lebron_bot_2//model/summary.json'
    fn_corpus = '/home/ivan/catkin_ws/src/lebron_bot_2//data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def infer(model: Model, fn_img: Path) -> None:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(recognized[0])
    return(recognized[0])
    print(f'Probability: {probability[0]}')


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='/home/ivan/catkin_ws/src/lebron_bot_2//data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()

def Mask():
    cap = cv2.VideoCapture("/dev/video2")
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    def Contrastador(imagen):

        # img = cv2.imread('imatext.png', IMREAD_GRAYSCALE)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imageArray=np.array(gray)
        fil= int(imageArray.shape[0])
        filmed = int(imageArray.shape[0]/2)
        col = int(imageArray.shape[1])
    
        columna=0
        flipedrArray=np.fliplr(imageArray)
        flipudArray=np.flipud(imageArray)
        for i, j in np.ndindex(imageArray.shape):
            if imageArray[i][j]>70:
                imageArray[i, j] = 255
            if imageArray[i][j]<70:
                imageArray[i, j] = 0
    
        for j in range(imageArray.shape[1]):
        
            if (imageArray[:,j]-imageArray[:,-j+3]).any()!=0:
                columna =j
                print(columna)
                break
        for j in range(imageArray.shape[1]):
        
            if (flipedrArray[:,j]-flipedrArray[:,j-3]).any()!=0:
                columna1 =j
                print(columna1)
                break

        for i in range(imageArray.shape[0]):
        
            if (imageArray[i,:]-imageArray[i-3,:]).any()!=0:
                fila =i
                print(fila)
                break
        for i in range(imageArray.shape[0]):
        
            if (flipudArray[i,:]-flipudArray[i-3,:]).any()!=0:
                fila1 =i
                print(fila1)
                break
    
        filaNew = int((fila)-20)
        fila1New = int((fila1)-20)
        rangefil=int(fil-fila1New)
        columnaNew=int(columna-20)
        columna1New=int(columna1-20)
        rangecol=int(col-columna1New)
    
        crop_img = imageArray[filaNew:rangefil, columnaNew:rangecol]
        kernel = np.ones((4, 4), np.uint8)
        imgMorph = cv2.erode(crop_img, kernel, iterations = 1)
    
        print("Mask")


       
    # lower_gray = np.array([0, 0, 0], np.uint8)
    # upper_gray = np.array([179, 50, 230], np.uint8)
    # mask_gray = cv2.inRange(imageArray, lower_gray, upper_gray)
    # img_res = cv2.bitwise_and(img, img, mask = mask_gray)
        cv2.imshow('Logo OpenCV',imgMorph)
        path = '/home/ivan/catkin_ws/src/lebron_bot_2/data'
        cv2.imwrite(os.path.join(path , 'word.png'), imgMorph)
        
    #cv2.imwrite('im.png', imageArray)
    
        t = cv2.waitKey(1)
    k=0
    while True:
        ret, frame = cap.read()
        if ret==False:break
        doc = frame
        cv2.imshow("Lector inteligente", frame)
        t = cv2.waitKey(1)
        if t==27:break
    

    Contrastador(doc)
    cap.release()
    cv2.destroyAllWindows()



def main():
    """Main function."""
    print("Main")
    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # train the model
    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

        # when in line mode, take care to have a whitespace in the char list
        char_list = loader.char_list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters and words
        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))

        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, decoder_type)
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        validate(model, loader, args.line_mode)

    # infer text on test image
    elif args.mode == 'infer':
        print()
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        PalabraReconocida = infer(model, args.img_file)
        print(PalabraReconocida)
        
    return(PalabraReconocida)
        



if __name__ == '__main__':
    Mask()
    main()
