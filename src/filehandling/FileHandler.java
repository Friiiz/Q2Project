package filehandling;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

public class FileHandler {

    private final ConcurrentHashMap<double[], Character> TRAINING_DATA;

    private final LinkedList<Thread> THREADS;

    private boolean allFilesLoaded;

    public FileHandler() {
        TRAINING_DATA = new ConcurrentHashMap<>();
        THREADS = new LinkedList<>();
        allFilesLoaded = false;
    }

    public void loadFiles() throws InterruptedException {
        System.out.println("Loading and compressing data...");
        File database = new File("C:\\Users\\Friiiz\\Documents\\NIST Handwritten Forms and Characters Database");
        for (File folder : Objects.requireNonNull(database.listFiles(), "No folders found in " + database.getAbsolutePath())) {
            Runnable fileLoader = () -> {
                for (File file : Objects.requireNonNull(new File(folder + "/train_" + folder.getName()).listFiles(), "No file found in " + folder.getAbsolutePath())) {
                    BufferedImage image;
                    try {
                        image = ImageIO.read(file);
                    } catch (IOException e) {
                        e.printStackTrace();
                        System.out.println("Thread " + Thread.currentThread().getName() + " could not read file " + file.getAbsolutePath() + ", continuing to load other files...");
                        continue;
                    }
                    char character = (char) Integer.parseInt(folder.getName(), 16);

                    byte[][] uncompressed = new byte[image.getWidth()][image.getHeight()];
                    double[] compressedImage = new double[1024];

                    for (int y = 0; y < image.getHeight(); y++) {
                        for (int x = 0; x < image.getWidth(); x++) {
                            uncompressed[x][y] = (byte) (image.getRGB(x, y) / -16777215.0d - 0.0000000596046473d);
                        }
                    }

                    int i = 0;
                    for (int y = 0; y < uncompressed.length - 4; y += 4) {
                        for (int x = 0; x < uncompressed[y].length - 4; x += 4) {
                            compressedImage[i] = (uncompressed[x][y] + uncompressed[x + 1][y] + uncompressed[x + 2][y] + uncompressed[x + 3][y] + uncompressed[x + 1][y + 1] + uncompressed[x + 2][y + 1] + uncompressed[x + 3][y + 1] + uncompressed[x + 1][y + 2] + uncompressed[x + 2][y + 2] + uncompressed[x + 3][y + 2] + uncompressed[x + 1][y + 3] + uncompressed[x + 2][y + 3] + uncompressed[x + 3][y + 3]) / 16.0d;
                            i++;
                        }
                        i++;
                    }

                    TRAINING_DATA.put(compressedImage, character);
                }
                System.out.println("File loader thread " + Thread.currentThread().getName() + " terminated.");
            };
            Thread fileLoaderThread = new Thread(fileLoader, String.valueOf((char) Integer.parseInt(folder.getName(), 16)));
            fileLoaderThread.start();
            System.out.println("File loader thread " + fileLoaderThread.getName() + " started.");
            THREADS.add(fileLoaderThread);
        }
        for (Thread thread : THREADS) {
            thread.join();
        }
        allFilesLoaded = true;
        System.out.println("Number of files loaded: " + TRAINING_DATA.size());
    }

    @Deprecated(since = "implementation multithreaded file loading", forRemoval = true)
    public void saveTrainingData() throws IOException {
        System.out.println("Saving compressed data...");

        File data = new File("data.txt");
        if (data.createNewFile()) {
            System.out.println("File created: " + data.getAbsolutePath());
        } else {
            System.out.println("File already exists.");
        }
    }

    public LinkedHashMap<double[], Character> getTrainingData() {
        if (!allFilesLoaded) throw new IllegalStateException("The files have not been fully loaded.");
        return new LinkedHashMap<>(TRAINING_DATA);
    }

}
