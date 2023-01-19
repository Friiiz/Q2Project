package gui;

import network.Neuron;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;

import java.util.LinkedList;
import java.util.List;

public class DrawingPane extends JPanel implements MouseListener, MouseMotionListener {

    boolean leftClickPressed;
    private final int DRAWING_RESOLUTION;
    private final float STROKE_WIDTH;
    private final LinkedList<LinkedList<Point2D.Double>> DRAWN_STROKES;
    private MouseEvent mousePositionEvent;

    public DrawingPane() {
        //setPreferredSize(new Dimension(width, height));
        addMouseListener(this);
        addMouseMotionListener(this);
        setFocusable(true);
        grabFocus();
        setFocusTraversalKeysEnabled(false);
        setBackground(Color.WHITE);
        DRAWING_RESOLUTION = 0;
        STROKE_WIDTH = 10;
        DRAWN_STROKES = new LinkedList<>();
        leftClickPressed = false;
    }

    public static final int MIN_MAX_NORMALIZATION = 1;
    public static final int Z_SCORE_NORMALIZATION = 2;

    public void regularizeImage(int regularizationAlgorithm) {
        final double PADDING = 50;
        final double X_SCALE_FACTOR = getWidth() - 2 * PADDING;
        final double Y_SCALE_FACTOR = getHeight() - 2 * PADDING;

        if(regularizationAlgorithm == MIN_MAX_NORMALIZATION) {
            //apply min-max normalization
            double maxX = DRAWN_STROKES.stream().mapToDouble(stroke -> stroke.stream().mapToDouble(point -> point.x).max().orElseThrow()).max().orElseThrow();
            double minX = DRAWN_STROKES.stream().mapToDouble(stroke -> stroke.stream().mapToDouble(point -> point.x).min().orElseThrow()).min().orElseThrow();
            double maxY = DRAWN_STROKES.stream().mapToDouble(stroke -> stroke.stream().mapToDouble(point -> point.y).max().orElseThrow()).max().orElseThrow();
            double minY = DRAWN_STROKES.stream().mapToDouble(stroke -> stroke.stream().mapToDouble(point -> point.y).min().orElseThrow()).min().orElseThrow();

            LinkedList<LinkedList<Point2D.Double>> drawnStrokesCopy = new LinkedList<>(DRAWN_STROKES);

            for (LinkedList<Point2D.Double> stroke : DRAWN_STROKES) {
                for (Point2D.Double point : stroke) {
                    point.setLocation(((point.x - minX) / (maxX - minX)) * X_SCALE_FACTOR + PADDING, ((point.y - minY) / (maxY - minY)) * Y_SCALE_FACTOR + PADDING);
                }
            }
        } else if (regularizationAlgorithm == Z_SCORE_NORMALIZATION) {
            double meanX = DRAWN_STROKES.stream().mapToDouble(stroke -> stroke.stream().mapToDouble(point -> point.x).sum()).sum() / DRAWN_STROKES.stream().mapToDouble(LinkedList::size).sum();
            double standardDeviationX = Math.sqrt(DRAWN_STROKES.stream().mapToDouble(stroke -> stroke.stream().mapToDouble(point -> (point.x - meanX) * (point.x - meanX)).sum()).sum());
            double meanY = DRAWN_STROKES.stream().mapToDouble(stroke -> stroke.stream().mapToDouble(point -> point.y).sum()).sum() / DRAWN_STROKES.stream().mapToDouble(LinkedList::size).sum();
            double standardDeviationY = Math.sqrt(DRAWN_STROKES.stream().mapToDouble(stroke -> stroke.stream().mapToDouble(point -> (point.y - meanY) * (point.y - meanY)).sum()).sum());

            for (LinkedList<Point2D.Double> stroke : DRAWN_STROKES) {
                for (Point2D.Double point : stroke) {
                    point.setLocation(((point.x - meanX) / standardDeviationX) * X_SCALE_FACTOR + PADDING, ((point.y - meanY) / standardDeviationY) * Y_SCALE_FACTOR + PADDING);
                }
            }
        } else {
            throw new IllegalArgumentException(regularizationAlgorithm + " is not a valid regularization algorithm.");
        }
    }

    @Override
    public void paintComponent(Graphics g) {
        //set up graphics object
        Graphics2D g2D = (Graphics2D) g;
        g2D.setStroke(new BasicStroke(STROKE_WIDTH));
        g2D.setColor(Color.BLACK);

        //reset canvas
        super.paintComponent(g2D);

        //convert points to paths
        LinkedList<Path2D.Double> strokes = new LinkedList<>();

        for (LinkedList<Point2D.Double> stroke : DRAWN_STROKES) {
            Path2D.Double path = new Path2D.Double();
            path.moveTo(stroke.getFirst().x, stroke.getFirst().y);

            for (int i = 1; i < stroke.size(); i++) {
                Point2D.Double point = stroke.get(i);
                path.lineTo(point.x, point.y);
            }
            strokes.add(path);
        }

        //draw paths
        for (Path2D.Double stroke : strokes) {
            g2D.draw(stroke);
        }
    }

    @Override
    public void mouseClicked(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e) {
        if (e.getButton() == MouseEvent.BUTTON1) {
            System.out.println("pressed");
            leftClickPressed = true;


            Runnable drawingRunnable = () -> {
                if(mousePositionEvent == null) {
                    mousePositionEvent = e;
                }

                //save initial mouse position and add it to current stroke
                int mouseX = mousePositionEvent.getX();
                int mouseY = mousePositionEvent.getY();
                DRAWN_STROKES.add(new LinkedList<>(List.of(new Point2D.Double(mousePositionEvent.getX(), mousePositionEvent.getY()))));

                System.out.println("thread created");

                //when holding add new point to current stroke whenever mouse moves by drawing resolution
                while (leftClickPressed) {
                    if (Point.distance(mousePositionEvent.getX(), mousePositionEvent.getY(), mouseX, mouseY) > DRAWING_RESOLUTION) {
                        mouseX = mousePositionEvent.getX();
                        mouseY = mousePositionEvent.getY();

                        DRAWN_STROKES.getLast().add(new Point2D.Double(mousePositionEvent.getX(), mousePositionEvent.getY()));
                    }
                    repaint();
                }
            };
            Thread drawingThread = new Thread(drawingRunnable, "drawing thread");
            drawingThread.start();
        } else if (e.getButton() == MouseEvent.BUTTON3) {
            if(!DRAWN_STROKES.isEmpty()) {
                //DRAWN_STROKES.removeLast();
            }
            regularizeImage(MIN_MAX_NORMALIZATION);
            repaint();
            mousePositionEvent = null;
            System.out.println("test");
        }
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        if (e.getButton() == MouseEvent.BUTTON1) {
            leftClickPressed = false;
            mousePositionEvent = null;
            System.out.println("released");
            System.out.println(DRAWN_STROKES.size());
        }
    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {
        leftClickPressed = false;
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        mousePositionEvent = e;
    }

    @Override
    public void mouseMoved(MouseEvent e) {

    }
}
