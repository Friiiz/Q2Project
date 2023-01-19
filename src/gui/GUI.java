package gui;

import javax.swing.*;
import java.awt.*;

public class GUI {

    JFrame frame;
    DrawingPane drawingPane;

    public static final int SCREEN_WIDTH = (int) Toolkit.getDefaultToolkit().getScreenSize().getWidth();
    public static final int SCREEN_HEIGHT = (int) Toolkit.getDefaultToolkit().getScreenSize().getHeight();

    public GUI() {
        this.frame = new JFrame();
        frame.setResizable(false);
        frame.setFocusTraversalKeysEnabled(false);
        frame.setBackground(Color.WHITE);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setTitle("Character Recognition AI");

        JPanel panel = new JPanel();
        panel.setBounds(frame.getContentPane().getBounds());
        frame.add(panel);

        GridBagLayout layout = new GridBagLayout();
        GridBagConstraints constraints = new GridBagConstraints();
        panel.setLayout(layout);

        constraints.gridx = 0;
        constraints.gridy = 0;
        constraints.gridheight = 2;
        constraints.fill = GridBagConstraints.BOTH;

        this.drawingPane = new DrawingPane(panel.getWidth() / 2, panel.getHeight());

        layout.setConstraints(drawingPane, constraints);

        panel.add(drawingPane);

        constraints.gridx = 1;
        constraints.gridy = 0;
        constraints.gridheight = 1;

        JLabel normalizedImage = new JLabel("test"); //new ImageIcon(new byte[1024])
        normalizedImage.setBackground(Color.CYAN);
        normalizedImage.setOpaque(true);
        normalizedImage.setHorizontalTextPosition(SwingConstants.CENTER);
        //normalizedImage.setPreferredSize(new Dimension(panel.getWidth() / 2, panel.getHeight() / 2));
        layout.setConstraints(normalizedImage, constraints);

        panel.add(normalizedImage);

        constraints.gridx = 1;
        constraints.gridy = 1;

        JLabel detectedCharacter = new JLabel("test2");
        detectedCharacter.setHorizontalTextPosition(SwingConstants.CENTER);
        detectedCharacter.setOpaque(true);
        detectedCharacter.setPreferredSize(new Dimension(panel.getWidth() / 2, panel.getHeight() / 2));

        layout.setConstraints(detectedCharacter, constraints);
        panel.add(detectedCharacter);

        frame.setBounds(SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2);
        frame.setVisible(true);
        panel.setVisible(true);
        drawingPane.setVisible(true);
    }
}
