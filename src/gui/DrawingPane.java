package gui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.util.LinkedList;
import java.util.List;

public class DrawingPane extends JPanel implements MouseListener {

    boolean leftClickPressed;
    private final int DRAWING_RESOLUTION;
    private final LinkedList<LinkedList<Point>> DRAWN_STROKES;

    public DrawingPane(int width, int height) {
        //setPreferredSize(new Dimension(width, height));
        addMouseListener(this);
        setFocusable(true);
        grabFocus();
        setFocusTraversalKeysEnabled(false);
        setBackground(Color.BLACK);
        DRAWING_RESOLUTION = 5;
        DRAWN_STROKES = new LinkedList<>();
        leftClickPressed = false;
    }

    @Override
    public void paintComponent(Graphics g) {
        //set up graphics object
        Graphics2D g2D = (Graphics2D) g;
        g2D.setStroke(new BasicStroke(0.5f));
        g2D.setColor(Color.BLACK);

        //reset canvas
        super.paintComponent(g2D);

        //convert points to paths
        LinkedList<Path2D.Double> strokes = new LinkedList<>();

        for (LinkedList<Point> stroke : DRAWN_STROKES) {
            strokes.add(new Path2D.Double());
            strokes.getLast().moveTo(stroke.getFirst().x, stroke.getFirst().y);
            for (int i = 1; i < stroke.size(); i++) {
                Point point = stroke.get(i);
                strokes.getLast().lineTo(point.x, point.y);
            }
        }

        //draw paths
        for (Path2D.Double stroke : strokes) {
            g2D.draw(stroke);
        }
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        if (e.getButton() == MouseEvent.BUTTON2) {
            DRAWN_STROKES.removeLast();
        }
    }

    @Override
    public void mousePressed(MouseEvent e) {
        if (e.getButton() == MouseEvent.BUTTON1) {
            leftClickPressed = true;
            Point mousePosition = e.getPoint();
            DRAWN_STROKES.add(new LinkedList<>(List.of(mousePosition)));

            if (Math.abs(mousePosition.x - e.getX()) >= DRAWING_RESOLUTION || Math.abs(mousePosition.y - e.getY()) >= DRAWING_RESOLUTION) {
                mousePosition = e.getPoint();
                DRAWN_STROKES.getLast().add(mousePosition);
                repaint();
            }
        }
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        if (e.getButton() == MouseEvent.BUTTON1) {
            leftClickPressed = false;
        }
    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }
}
