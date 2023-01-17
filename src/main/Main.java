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

import ann.Network;

public class Main {
    public static void main(String[] args) {
        Network network = new Network(0.01, 1024, 62);
        network.train();
    }
}
