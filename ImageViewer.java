import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.nio.Buffer;

public class ImageViewer {
    private final JFrame frame;
    private JLabel label;

    public ImageViewer(String s) {
        frame = new JFrame(s);
        label = new JLabel();
        frame.add(label);
        frame.setResizable(false);
    }

    public void show(String filePath, int width, int height) {
        frame.remove(label);
        ImageIcon image = new ImageIcon(filePath);
        label = new JLabel(
                new ImageIcon(image.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT)));
        frame.add(label);
        frame.pack();
        frame.validate();
        frame.repaint();
    }

    public void show(ImageIcon image, int width, int height) {
        frame.remove(label);
        label = new JLabel(
                new ImageIcon(image.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT)));
        frame.add(label);
        frame.pack();
        frame.validate();
        frame.repaint();
    }

    public void show(String filePath) throws IOException {
        show(pathToImage(filePath));
    }

    public void show(BufferedImage image) {
        ImageIcon icon = new ImageIcon(image);
        show(icon, icon.getIconWidth(), icon.getIconHeight());
    }

    public void show(String filePath, double scale) throws IOException {
        show(pathToImage(filePath), scale);
    }

    public void show(BufferedImage image, double scale) {
        ImageIcon icon = new ImageIcon(image);
        show(icon, (int) (icon.getIconWidth() * scale), (int) (icon.getIconHeight() * scale));
    }

    public void setVisible(boolean visible) {
        frame.setVisible(visible);
        frame.repaint();
    }

    static int[][][] imageToList(BufferedImage image, boolean isColor) {
        final byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        int alphaOffset = image.getAlphaRaster() != null ? 1 : 0;
        final int width = image.getWidth();
        final int height = image.getHeight();
        int[][][] result = new int[width][height][];
        for (int i = 0, row = 0, col = 0; i < pixels.length - 2 - alphaOffset; i += 3 + alphaOffset) {
            int red = pixels[i + 2 + alphaOffset] & 0xff;
            int green = pixels[i + 1 + alphaOffset] & 0xff;
            int blue = pixels[i + alphaOffset] & 0xff;
            if (isColor) {
                result[row][col] = new int[] { red, green, blue };
            } else {
                result[row][col] = new int[] { (red + green + blue) / 3 };
            }
            col++;
            if (col == height) {
                row++;
                col = 0;
            }
        }
        return result;
    }

    static BufferedImage listToImage(int[][][] dubs) {
        boolean isColor = dubs[0][0].length == 3;
        BufferedImage out = new BufferedImage(dubs.length, dubs[0].length, BufferedImage.TYPE_INT_RGB);
        int[] outPixels = ((DataBufferInt) out.getRaster().getDataBuffer()).getData();
        int red, green, blue;
        for (int r = 0; r < dubs.length; r++) {
            for (int c = 0; c < dubs[0].length; c++) {
                if (isColor) {
                    red = dubs[r][c][0] << 16;
                    green = dubs[r][c][1] << 8;
                    blue = dubs[r][c][2];
                } else {
                    red = dubs[r][c][0] << 16;
                    green = dubs[r][c][0] << 8;
                    blue = dubs[r][c][0];
                }
                outPixels[r * dubs[0].length + c] = red | green | blue;
            }
        }
        return out;
    }

    static BufferedImage pathToImage(String filePath) throws IOException {
        return ImageIO.read(new File(filePath));
    }
    
    static void imageToPath(BufferedImage image, String filePath) throws IOException{
        ImageIO.write(image, "jpg", new File(filePath));
    }
}
