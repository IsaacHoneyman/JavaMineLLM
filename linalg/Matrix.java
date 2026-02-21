package linalg;

public final class Matrix {
    static final int BLOCK_SIZE = 16;

    public final int rows;
    public final int columns;

    private final float[] data;

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;

        data = new float[rows * columns];
    }

    public Matrix(int rows, int columns, float[] data) {
        if (data.length != rows * columns)
            throw new IllegalArgumentException("Dimension mismatch");

        this.rows = rows;
        this.columns = columns;
        this.data = data.clone();
    }

    // --- Basic Functions ---

    public float get(int row, int col) {
        return data[row * columns + col];
    }

    public void set(int row, int col, float value) {
        data[row * columns + col] = value;
    }

    public void print(int maxPrecision) {
        int[] maxColLen = new int[columns];

        for (int x = 0; x < columns; x++) {
            int max = 0;
            for (int y = 0; y < rows; y++) {
                float val = UtilMath.round(get(y, x), maxPrecision);
                max = Math.max(String.valueOf(val).length(), max);
            }
            maxColLen[x] = max;
        }

        var sb = new StringBuilder();
        for (int y = 0; y < rows; y++) {
            sb.append("[");
            for (int x = 0; x < columns; x++) {
                float val = UtilMath.round(get(y, x), maxPrecision);
                sb.append(String.format("%-" + maxColLen[x] + "s", val)).append(x == columns - 1 ? "" : ", ");
            }
            sb.append("]\n");
        }

        System.out.print(sb);
    }

    public void print() {
        print(4);
    }

    public void zero() {
        java.util.Arrays.fill(data, 0f);
    }

    public void randomise(float min, float max) {
        for (int i = 0; i < data.length; i++) {
            data[i] = UtilMath.rng.nextFloat(min, max);
        }
    }

    // Linear Algebra ---

    /**
     * Multiples two matrices via IKJ-ordered for cache locality.
     * 
     * @param a The left-hand matrix.
     * @param b The right-hand matrix.
     * @return A new Matrix representing the dot product of a and b.
     * @throws IllegalArgumentException if dimensions are incompatible.
     */
    public static Matrix dot(Matrix a, Matrix b) {
        if (a.columns != b.rows)
            throw new IllegalArgumentException("Dimension Mismatch");
        Matrix result = new Matrix(a.rows, b.columns);

        for (int i = 0; i < a.rows; i++) {
            int resRowOffset = i * result.columns;
            int aRowOffset = i * a.columns;
            for (int k = 0; k < a.columns; k++) {
                float aVal = a.data[aRowOffset + k];
                int bRowOffset = k * b.columns;
                for (int j = 0; j < b.columns; j++) {
                    result.data[resRowOffset + j] += aVal * b.data[bRowOffset + j];
                }
            }
        }

        return result;
    }

    public Matrix dot(Matrix other) {
        return dot(this, other);
    }

    public Matrix transpose() {
        Matrix result = new Matrix(columns, rows);
        for (int i = 0; i < rows; i += BLOCK_SIZE) {
            for (int j = 0; j < columns; j += BLOCK_SIZE) {
                int yMax = Math.min(i + BLOCK_SIZE, rows);
                int xMax = Math.min(j + BLOCK_SIZE, columns);

                for (int y = i; y < yMax; y++) {
                    int rowOffset = y * columns;
                    for (int x = j; x < xMax; x++) {
                        result.data[x * rows + y] = this.data[rowOffset + x];
                    }
                }
            }
        }
        return result;
    }

    // --- Element Wise ---

    public static Matrix add(Matrix a, Matrix b) {
        if (a.columns != b.columns || a.rows != b.rows)
            throw new IllegalArgumentException("Dimension Mismatch");

        Matrix result = new Matrix(a.rows, a.columns);

        for (int i = 0; i < a.data.length; i++) {
            result.data[i] = a.data[i] + b.data[i];
        }

        return result;
    }

    public Matrix add(Matrix other) {
        return add(this, other);
    }

    public Matrix add(float scalar) {
        Matrix result = new Matrix(rows, columns);

        for (int i = 0; i < data.length; i++) {
            result.data[i] = data[i] + scalar;
        }

        return result;
    }

    public static Matrix subtract(Matrix a, Matrix b) {
        if (a.columns != b.columns || a.rows != b.rows)
            throw new IllegalArgumentException("Dimension Mismatch");

        Matrix result = new Matrix(a.rows, a.columns);

        for (int i = 0; i < a.data.length; i++) {
            result.data[i] = a.data[i] - b.data[i];
        }

        return result;
    }

    public Matrix subtract(Matrix other) {
        return subtract(this, other);
    }

    public Matrix multiply(float scalar) {
        Matrix result = new Matrix(rows, columns);

        for (int i = 0; i < data.length; i++) {
            result.data[i] = data[i] * scalar;
        }

        return result;
    }

    public static Matrix hadamard(Matrix a, Matrix b) {
        if (a.columns != b.columns || a.rows != b.rows)
            throw new IllegalArgumentException("Dimension Mismatch");

        Matrix result = new Matrix(a.rows, a.columns);

        for (int i = 0; i < a.data.length; i++) {
            result.data[i] = a.data[i] * b.data[i];
        }

        return result;
    }

    public Matrix hadamard(Matrix other) {
        return hadamard(this, other);
    }

    // --- Reductions / Activations ---

    public Matrix gelu() {
        Matrix result = new Matrix(rows, columns);

        for (int i = 0; i < data.length; i++) {
            float x = data[i];
            float inner = UtilMath.SQRT_2_OVER_PI * (x + UtilMath.GELU_APROX * (x * x * x));
            result.data[i] = (float) (0.5f * x * (1.0f + Math.tanh(inner)));
        }

        return result;
    }

    /**
     * Applies the Softmax function row-wise to convert logits into probabilities.
     * Uses the max-subtraction trick for numerical stability.
     * 
     * @return A new Matrix containing the probability distributions.
     */
    public Matrix softmax() {
        Matrix result = new Matrix(rows, columns);

        for (int y = 0; y < rows; y++) {
            int rowOffset = y * columns;

            float max = data[rowOffset];
            for (int x = 1; x < columns; x++) {
                if (data[rowOffset + x] > max) {
                    max = data[rowOffset + x];
                }
            }

            float sum = 0f;
            for (int x = 0; x < columns; x++) {
                result.data[rowOffset + x] = (float) Math.exp(data[rowOffset + x] - max);
                sum += result.data[rowOffset + x];
            }

            for (int j = 0; j < columns; j++) {
                result.data[rowOffset + j] /= sum;
            }
        }

        return result;
    }

    /**
     * Applies Layer Normalization row-wise to stabilize network activations.
     * * @return A new normalized Matrix.
     */
    public Matrix rmsNorm() {
        Matrix result = new Matrix(rows, columns);

        for (int y = 0; y < rows; y++) {
            int rowOffset = y * columns;

            float squareSum = 0f;
            for (int x = 0; x < columns; x++) {
                float val = data[rowOffset + x];
                squareSum += val * val;
            }
            float meanSquare = squareSum / columns;

            float invRms = (float) (1.0 / Math.sqrt(meanSquare + UtilMath.EPSILON_5));

            for (int x = 0; x < columns; x++) {
                result.data[rowOffset + x] = data[rowOffset + x] * invRms;
            }
        }
        return result;
    }

    // --- Backward Functions ---

    /**
     * Calculates the backward pass (derivative) of the GELU activation.
     * * @param gradOutput The error gradient flowing backward from the next layer.
     * 
     * @return A new Matrix containing the gradients to pass further back.
     */
    public Matrix geluBackward(Matrix gradOutput) {
        Matrix result = new Matrix(rows, columns);

        for (int i = 0; i < data.length; i++) {
            float x = data[i];

            float inner = UtilMath.SQRT_2_OVER_PI * (x + UtilMath.GELU_APROX * (x * x * x));
            float cdf = 0.5f * (1.0f + (float) Math.tanh(inner)); // cumulative distribution at x
            float pdf = UtilMath.INV_SQRT_2_OVER_PI * (float) Math.exp(-0.5f * x * x); // prob density at x

            result.data[i] = (cdf + x * pdf) * gradOutput.data[i];
        }

        return result;
    }

    /**
     * Calculates the backward pass of the Softmax activation.
     * Note: This method must be called on the MATRIX OF PROBABILITIES (the forward
     * output),
     * not the original raw logits.
     * * @param gradOutput The error gradient flowing backward.
     * 
     * @return A new Matrix containing the gradients for the raw logits.
     */
    public Matrix softmaxBackward(Matrix gradOutput) {
        Matrix result = new Matrix(rows, columns);

        for (int y = 0; y < rows; y++) {
            int rowOffset = y * columns;

            float dotP = 0f;
            for (int x = 0; x < columns; x++) {
                dotP += this.data[rowOffset + x] * gradOutput.data[rowOffset + x];
            }

            for (int x = 0; x < columns; x++) {
                float prob = this.data[rowOffset + x];
                result.data[rowOffset + x] = prob * (gradOutput.data[rowOffset + x] - dotP);
            }
        }

        return result;
    }

    /**
     * Calculates the backward pass of the RMSNorm activation.
     * 
     * @param gradOutput The error gradient flowing backward from the next layer.
     * @return A new Matrix containing the gradients to pass further back.
     */
    public Matrix rmsNormBackward(Matrix gradOutput) {
        Matrix result = new Matrix(rows, columns);

        for (int y = 0; y < rows; y++) {
            int rowOffset = y * columns;

            float dotP = 0f;
            float squareSum = 0f;
            for (int x = 0; x < columns; x++) {
                squareSum += data[rowOffset + x] * data[rowOffset + x];
                dotP += this.data[rowOffset + x] * gradOutput.data[rowOffset + x];

            }
            float invRms = (float) (1.0 / Math.sqrt((squareSum / columns) + UtilMath.EPSILON_5));
            float scale = (dotP * invRms * invRms) / columns;

            for (int x = 0; x < columns; x++) {
                result.data[rowOffset + x] = invRms * (gradOutput.data[rowOffset + x] - 
                    (data[rowOffset + x] * scale));
            }
        }

        return result;
    }

    // --- Other ---

    /**
     * Calculates the Cross-Entropy Loss between predictions and true targets.
     * * @param predictions The output from the Softmax layer (probabilities).
     * 
     * @param targets The one-hot encoded true values.
     * @return A scalar float representing the total network error.
     */
    public static float crossEntropyLoss(Matrix preds, Matrix targs) {
        if (preds.rows != targs.rows || preds.columns != targs.columns)
            throw new IllegalArgumentException("Dimension Mismatch");

        float loss = 0f;
        for (int i = 0; i < preds.data.length; i++) {
            if (targs.data[i] > 0f) {
                loss -= targs.data[i] * (float) Math.log(preds.data[i] + UtilMath.EPSILON_5);
            }
        }

        return loss / preds.rows;
    }
}
