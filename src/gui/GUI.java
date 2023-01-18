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
        frame.setBounds(SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2);
        frame.setResizable(false);
        frame.setFocusTraversalKeysEnabled(false);
        frame.setBackground(Color.BLACK);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setTitle("Character Recognition AI");
        this.drawingPane = new DrawingPane(frame.getContentPane().getX(), frame.getContentPane().getY(), frame.getContentPane().getWidth() / 2, frame.getContentPane().getHeight());
        frame.add(drawingPane);
        drawingPane.setVisible(true);
        frame.setVisible(true);
    }
}
