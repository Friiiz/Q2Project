package gui;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class GUI {

    JFrame frame;
    DrawingPane drawingPane;

    public static final int SCREEN_WIDTH = (int) Toolkit.getDefaultToolkit().getScreenSize().getWidth();
    public static final int SCREEN_HEIGHT = (int) Toolkit.getDefaultToolkit().getScreenSize().getHeight();

    public GUI() {
        this.frame = new JFrame();
        //frame.setResizable(false);
        frame.setFocusTraversalKeysEnabled(false);
        frame.setBackground(Color.WHITE);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setTitle("Character Recognition AI");
        //FIXME: panel's height doesn't match width
        frame.setBounds(SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_WIDTH / 4);
        Insets insets = frame.getInsets();
        frame.setBounds(frame.getX() + (insets.left + insets.right) / 2, frame.getY() + (insets.top + insets.bottom) / 2, frame.getWidth() + (insets.left + insets.right) / 2, frame.getHeight() +  (insets.top + insets.bottom) / 2);


        JPanel panel = new JPanel();
        panel.setPreferredSize(frame.getContentPane().getSize());
        panel.setBackground(Color.LIGHT_GRAY);

        GridBagLayout layout = new GridBagLayout();
        GridBagConstraints constraints = new GridBagConstraints();
        panel.setLayout(layout);

        constraints.gridx = 0;
        constraints.gridy = 0;
        constraints.gridheight = 2;
        constraints.weightx = 0.5;
        constraints.weighty = 0.5;
        constraints.fill = GridBagConstraints.BOTH;

        this.drawingPane = new DrawingPane();

        panel.add(drawingPane, constraints);

        constraints.gridx = 1;
        constraints.gridy = 0;
        constraints.gridheight = 1;
        constraints.weightx = 0.5;
        constraints.weighty = 0.5;

        JLabel regularizedImage = new JLabel(new ImageIcon(new BufferedImage(1, 1, BufferedImage.TYPE_BYTE_GRAY))); //new ImageIcon(new byte[1024])
        regularizedImage.setBackground(Color.LIGHT_GRAY);
        regularizedImage.setOpaque(true);
        regularizedImage.setHorizontalTextPosition(SwingConstants.CENTER);
        regularizedImage.setVerticalTextPosition(SwingConstants.CENTER);

        panel.add(regularizedImage, constraints);

        constraints.gridx = 1;
        constraints.gridy = 1;
        constraints.gridheight = 1;
        constraints.weightx = 0.5;
        constraints.weighty = 0.5;

        JLabel detectedCharacter = new JLabel("Recognized as", SwingConstants.CENTER);
        detectedCharacter.setHorizontalTextPosition(SwingConstants.CENTER);
        detectedCharacter.setVerticalTextPosition(SwingConstants.CENTER);
        detectedCharacter.setBackground(Color.LIGHT_GRAY);
        detectedCharacter.setOpaque(true);

        panel.add(detectedCharacter, constraints);

        frame.add(panel);
        //frame.pack();
        frame.setVisible(true);
    }
}
