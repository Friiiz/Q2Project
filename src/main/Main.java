/*
https://en.wikipedia.org/wiki/Machine_learning
https://en.wikipedia.org/wiki/Deep_learning
https://en.wikipedia.org/wiki/Genetic_algorithm
https://en.wikipedia.org/wiki/Artificial_neural_network
https://en.wikipedia.org/wiki/Computational_learning_theory
https://en.wikipedia.org/wiki/Statistical_learning_theory
https://en.wikipedia.org/wiki/Supervised_learning
https://en.wikipedia.org/wiki/Backtracking
 */

package main;

import filehandling.FileHandler;
import gui.GUI;
import network.Network;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Main {

    public static Network NETWORK = new Network(0.01, 100, 1024, 62);
    public static final FileHandler FILE_HANDLER = new FileHandler();
    public static final GUI GUI = new GUI();
    public static void main(String[] args) {
        //load the network from a file if it has been saved before otherwise train it
        if(Files.exists(Paths.get("network.ser"))) {
            try (FileInputStream fileInputStream = new FileInputStream("network.ser")) {
                ObjectInputStream inputStream = new ObjectInputStream(fileInputStream);
                NETWORK = (Network) inputStream.readObject();
                inputStream.close();
            } catch (IOException | ClassNotFoundException e) {
                throw new RuntimeException(e);
            }
        } else {
            NETWORK.train();
        }

        //NETWORK.test();
    }
}
