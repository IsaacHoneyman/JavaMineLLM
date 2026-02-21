import linalg.Matrix;

public class Main
{
    public static void main(String[] args) {
        Matrix ma = new Matrix(2, 2, new float[] {1, 2, 3, 4});
        Matrix mb = new Matrix(2, 2, new float[] {4, 3, 2, 1});

        Matrix mc = ma.transpose();
        ma.print();
        mb.print();
        mc.print();
    }
}