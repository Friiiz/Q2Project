package filehandling;

import javax.imageio.ImageIO;
import java.awt.*;
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

    public static final int BILINEAR_INTERPOLATION = 0;
    public static final int WEIGHTED_BILINEAR_INTERPOLATION = 1;
    public static final int LANCZOS_RESAMPLING = 2;


    public void loadFiles() throws InterruptedException {
        System.out.println("Loading and compressing data...");

        //iterating over all sub-folders of the database folder
        File database = new File("C:\\Users\\Friiiz\\Documents\\NIST Handwritten Forms and Characters Database");
        for (File folder : Objects.requireNonNull(database.listFiles(), "No folders found in " + database.getAbsolutePath())) {

            //creating a thread for each sub-folder that loads all the images
            Runnable fileLoader = () -> {
                for (File file : Objects.requireNonNull(new File(folder + "/train_" + folder.getName()).listFiles(), "No file found in " + folder.getAbsolutePath())) {

                    //read image
                    BufferedImage image;
                    try {
                        image = ImageIO.read(file);
                    } catch (IOException e) {
                        e.printStackTrace();
                        System.out.println("Thread " + Thread.currentThread().getName() + " could not read file " + file.getAbsolutePath() + ", continuing to load other files...");
                        continue;
                    }

                    //get character from hex code in folder name
                    char character = (char) Integer.parseInt(folder.getName(), 16);

                    TRAINING_DATA.put(getCompressedImage(image, WEIGHTED_BILINEAR_INTERPOLATION), character);
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

    public LinkedHashMap<double[], Character> getTrainingData() {
        if (!allFilesLoaded) throw new IllegalStateException("The files have not been fully loaded.");
        return new LinkedHashMap<>(TRAINING_DATA);
    }

    public double[] getCompressedImage(BufferedImage image, int downscalingAlgorithm) {
        final int IMAGE_RESOLUTION = 32;
        final double SCALE_FACTOR = (double) Math.max(image.getHeight(), image.getWidth()) / IMAGE_RESOLUTION; //TODO: not nice; rather make sure images from panel are square

        final boolean[] imageSaved = {false};

        //convert image to byte array
        byte[][] uncompressed = new byte[image.getWidth()][image.getHeight()];

        int maxX = 0;
        int minX = image.getWidth();
        int maxY = 0;
        int minY = image.getHeight();
        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                int pixel = (int) Math.round(image.getRGB(x, y) / -16777215.0d - 0.0000000596046473d);
                uncompressed[x][y] = (byte) pixel;
                if(pixel != 0) {
                    if(x > maxX) maxX = x;
                    if(x < minX) minX = x;
                    if(y > maxY) maxY = y;
                    if(y < minY) minY = y;
                }
            }
        }

        //BufferedImage tempBufferedImage = new BufferedImage(IMAGE_RESOLUTION, IMAGE_RESOLUTION, BufferedImage.TYPE_BYTE_GRAY);
        //Graphics2D tempGraphics = tempBufferedImage.createGraphics();
        //BufferedImage subImage = image.getSubimage(minX, minY, image.getWidth() - maxX, image.getHeight() - maxY);
        //Image tempImage = subImage.getScaledInstance(IMAGE_RESOLUTION, IMAGE_RESOLUTION, Image.SCALE_SMOOTH);
        //tempGraphics.drawImage(tempImage, 0, 0, null);
        //tempGraphics.dispose();
//
        //image = tempBufferedImage;

        //compress image to specified resolution
        double[] compressedImage = new double[IMAGE_RESOLUTION * IMAGE_RESOLUTION];

        switch(downscalingAlgorithm) {
            case BILINEAR_INTERPOLATION -> {
                int i = 0;
                for (int y = 0; y < uncompressed.length - SCALE_FACTOR; y += SCALE_FACTOR) {
                    for (int x = 0; x < uncompressed[y].length - SCALE_FACTOR; x += SCALE_FACTOR) {
                        int sum = 0;
                        for (int x1 = 0; x1 < SCALE_FACTOR; x1++) {
                            for (int y1 = 0; y1 < SCALE_FACTOR; y1++) {
                                sum += uncompressed[x + x1][y + y1];
                            }
                        }
                        compressedImage[i] = sum / (SCALE_FACTOR * SCALE_FACTOR);
                        //compressedImage[i] = (uncompressed[x][y] + uncompressed[x + 1][y] + uncompressed[x + 2][y] + uncompressed[x + 3][y] + uncompressed[x + 1][y + 1] + uncompressed[x + 2][y + 1] + uncompressed[x + 3][y + 1] + uncompressed[x + 1][y + 2] + uncompressed[x + 2][y + 2] + uncompressed[x + 3][y + 2] + uncompressed[x + 1][y + 3] + uncompressed[x + 2][y + 3] + uncompressed[x + 3][y + 3]) / 16.0d;
                        i++;
                    }
                    i++;
                }
            }

            case WEIGHTED_BILINEAR_INTERPOLATION -> {
                int i = 0;
                double a = 2.6339157935;
                double b = 2.5;
                for (int y = 0; y < uncompressed.length - SCALE_FACTOR; y += SCALE_FACTOR) {
                    for (int x = 0; x < uncompressed[y].length - SCALE_FACTOR; x += SCALE_FACTOR) {
                        int sum = 0;
                        for (int x1 = 0; x1 < SCALE_FACTOR; x1++) {
                            for (int y1 = 0; y1 < SCALE_FACTOR; y1++) {
                                double pixelPosition = (x1 + y1) / 2.0d;
                                sum += uncompressed[x + x1][y + y1] * ((3 / (1 + Math.exp(a * (pixelPosition - b)))) * (1 - 1 / (1 + Math.exp(a * (pixelPosition - b)))) * 2);
                            }
                        }
                        compressedImage[i] = sum / (SCALE_FACTOR * SCALE_FACTOR);
                        //compressedImage[i] = (uncompressed[x][y] + uncompressed[x + 1][y] + uncompressed[x + 2][y] + uncompressed[x + 3][y] + uncompressed[x + 1][y + 1] + uncompressed[x + 2][y + 1] + uncompressed[x + 3][y + 1] + uncompressed[x + 1][y + 2] + uncompressed[x + 2][y + 2] + uncompressed[x + 3][y + 2] + uncompressed[x + 1][y + 3] + uncompressed[x + 2][y + 3] + uncompressed[x + 3][y + 3]) / 16.0d;
                        i++;
                    }
                    i++;
                }
            }

            case LANCZOS_RESAMPLING -> {
                //TODO: try this ↑
            }

            default -> throw new IllegalStateException("Unexpected value: " + downscalingAlgorithm);
        }

        //if(!imageSaved[0]) {
        //    try {
        //        File outputFile = new File("saved.png");
        //        ImageIO.write(image, "png", outputFile);
        //        imageSaved[0] = true;
        //    } catch (IOException e) {
        //        e.printStackTrace();
        //    }
        //}

        return compressedImage;
    }

}
