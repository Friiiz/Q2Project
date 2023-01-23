package gui;

import filehandling.FileHandler;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

import static gui.DrawingPane.MIN_MAX_NORMALIZATION;
import static main.Main.FILE_HANDLER;
import static main.Main.NETWORK;

public class GUI {

    JFrame frame;
    DrawingPane drawingPane;

    private JLabel regularizedImage;
    private JLabel detectedCharacterLabel;

    public GUI() {
        regularizedImage = new JLabel(new ImageIcon(new BufferedImage(1, 1, BufferedImage.TYPE_BYTE_GRAY)));
        detectedCharacterLabel = new JLabel("Recognized as", SwingConstants.CENTER);

        this.frame = new JFrame();
        frame.setResizable(false);
        frame.setFocusTraversalKeysEnabled(false);
        frame.setBackground(Color.WHITE);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setTitle("Character Recognition AI");
        //FIXME: panel's height doesn't match width


        GridBagLayout layout = new GridBagLayout();
        GridBagConstraints constraints = new GridBagConstraints();
        frame.setLayout(layout);

        constraints.gridx = 0;
        constraints.gridy = 0;
        constraints.gridheight = 2;
        constraints.weightx = 0.5;
        constraints.weighty = 0.5;
        constraints.fill = GridBagConstraints.BOTH;

        this.drawingPane = new DrawingPane();

        frame.add(drawingPane, constraints);

        constraints.gridx = 1;
        constraints.gridy = 0;
        constraints.gridheight = 1;
        constraints.weightx = 0.5;
        constraints.weighty = 0.5;

        regularizedImage.setBackground(Color.LIGHT_GRAY);
        regularizedImage.setOpaque(true);
        regularizedImage.setHorizontalTextPosition(SwingConstants.CENTER);
        regularizedImage.setVerticalTextPosition(SwingConstants.CENTER);
        regularizedImage.setPreferredSize(new Dimension(240, 240));

        frame.add(regularizedImage, constraints);

        constraints.gridx = 1;
        constraints.gridy = 1;
        constraints.gridheight = 1;
        constraints.weightx = 0.5;
        constraints.weighty = 0.5;

        detectedCharacterLabel.setHorizontalTextPosition(SwingConstants.CENTER);
        detectedCharacterLabel.setVerticalTextPosition(SwingConstants.CENTER);
        detectedCharacterLabel.setBackground(Color.LIGHT_GRAY);
        detectedCharacterLabel.setOpaque(true);
        detectedCharacterLabel.setPreferredSize(new Dimension(240, 240));

        frame.add(detectedCharacterLabel, constraints);

        drawingPane.setPreferredSize(new Dimension(480, 480));
        drawingPane.setSize(drawingPane.getPreferredSize());
        frame.pack();
        frame.setVisible(true);
    }

    /**
     * Updates the GUI to show the info for the current input on the drawing panel.
     */
    public void updatePrediction() {
        BufferedImage regularizedInputImage = drawingPane.getRegularizedImage(MIN_MAX_NORMALIZATION);
        if(regularizedInputImage != null) {
            regularizedImage.setIcon(new ImageIcon(regularizedInputImage.getScaledInstance(32, 32, Image.SCALE_SMOOTH).getScaledInstance(regularizedImage.getWidth(), regularizedImage.getHeight(), Image.SCALE_SMOOTH)));

            char detectedCharacter = NETWORK.evaluate(FILE_HANDLER.getCompressedImage(regularizedInputImage, FileHandler.WEIGHTED_BILINEAR_INTERPOLATION)).getKey();
            double certainty = NETWORK.evaluate(FILE_HANDLER.getCompressedImage(regularizedInputImage, FileHandler.WEIGHTED_BILINEAR_INTERPOLATION)).getValue();

            detectedCharacterLabel.setText("Recognized as: " + detectedCharacter + " (" + Math.round(certainty * 10000) / 100 + "%)");
        } else {
            regularizedImage.setIcon(null);
            detectedCharacterLabel.setText(null);
        }
    }
}
