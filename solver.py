import time

from metrics import cnr, cv
from nlm import NLMeans
from tqdm import tqdm
from utils import *


def start():
    if args.single:
        # single image processing.
        activateDenoiser(
            NLMeans(),
            "denoised.png",
            config["SINGLE_IMAGE"],
            config["SAVE_PATH"],
            os.path.join(
                config["SAVE_PATH"], f"result_{args.h}_{args.small_window_size}_{args.big_window_size}.png"
            )
        )
        return

    if args.process_all:
        # read from config.
        slicedDataPath, savingPath, savingFigPath = \
            config["DATA_PATH"], config["SAVE_PATH"], config["SAVE_FIG_PATH"]

        logger.info(f" starting to process images:\n"
                    f"data path: {slicedDataPath}\n"
                    f"saving path: {savingPath}\n"
                    f"saving plots path: {savingFigPath}\n"
                    )

        # process all the data.
        processAllDataset(slicedDataPath, savingPath, savingFigPath)
        return


def processAllDataset(dataPath="", saveToPath="", saveFigPath=""):
    processedImageCounter = 0

    startTime = time.time()
    denoiser = NLMeans()
    for p in os.listdir(dataPath):
        for imageName in tqdm(os.listdir(os.path.join(dataPath, p)), desc=f"processing directory:{p}"):
            activateDenoiser(
                denoiser,
                imageName,
                os.path.join(dataPath, p, imageName),
                saveToPath,
                os.path.join(saveFigPath, imageName)
            )
            processedImageCounter += 1

    endTime = time.time()
    logger.info(f"{processedImageCounter} images processed. time took: {endTime - startTime}")


def activateDenoiser(denoiser, imageSaveName="", imagePath="", savePath="", saveFigPath=""):
    image = readImage(imagePath)

    processedImage = solveWithTimer(
        denoiser,
        image,
        h=args.h,
        small_window=args.small_window_size,
        big_window=args.big_window_size,
    )

    if args.save:
        # save path already embedded with the image name.
        saveImage(processedImage, os.path.join(savePath, imageSaveName))

    plotWithMetrics(
        originalImage=image,
        processedImage=processedImage,
        plot=args.plot,
        saveFigPath=saveFigPath  # saveFigPath already embedded with the image name.
    )


def plotWithMetrics(originalImage, processedImage, saveFigPath, plot, defaultFigureSize=(15, 15)):
    cnrOriginal = cnr(
        image=originalImage,
        signal_mask=patchImage(originalImage, 85, 135, 85, 135),
        background_mask=patchImage(originalImage, 0, 50, 0, 50)
    )
    cvOriginal = cv(
        image=originalImage,
        mask=patchImage(originalImage, 85, 135, 85, 135)
    )

    cnrDenoised = cnr(
        image=processedImage,
        signal_mask=patchImage(processedImage, 85, 135, 85, 135),
        background_mask=patchImage(processedImage, 0, 50, 0, 50)
    )
    cvDenoised = cv(
        image=processedImage,
        mask=patchImage(originalImage, 85, 135, 85, 135)
    )

    logger.info(
        f"\nCNRs: original:{cnrOriginal:.2f} denoised:{cnrDenoised:.2f}\n"
        f"CVs: original:{cvOriginal:.2f} denoised:{cvDenoised:.2f}"
    )

    # matplotlib settings.
    plt.figure(figsize=defaultFigureSize)
    showGray(
        originalImage, 1, "Original Image"
    )
    showGray(
        processedImage,
        2,
        f"CNRs: original: {cnrOriginal:.2f} denoised:{cnrDenoised:.2f}\n"
        f"CVs: original: {cvOriginal:.2f} denoised:{cvDenoised:.2f}\n"
        f"NLM parameters: h: {args.h} small-win: {args.small_window_size} big-win: {args.big_window_size}"
    )

    if saveFigPath is not None and saveFigPath != "":
        plt.savefig(saveFigPath, bbox_inches="tight", dpi=300)
        logger.info(f"plot saved to {saveFigPath}")

    if plot:
        plt.show()


def solveWithTimer(denoiser: NLMeans, image, h=15, small_window=3, big_window=21) -> np.array:
    startTime = time.time()
    denoisedImage = denoiser.solve(image.copy(), h, small_window, big_window)
    endTime = time.time()
    logger.info(f"time took to process image: {endTime - startTime:.2f}s")
    return denoisedImage


# setup env
setupPlotter()
# CLI args.
args = parseArgs()
# readYamlConfig uses the default config path related to the project dir.
config = readYamlConfig()
if __name__ == "__main__":
    # prepare result dir.
    os.makedirs(config["SAVE_PATH"], exist_ok=True)
    os.makedirs(config["SAVE_FIG_PATH"], exist_ok=True)

    # start image-processing using the configurations.
    start()
