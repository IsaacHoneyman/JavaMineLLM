package linalg;

import java.util.Random;

public final class UtilMath {
    public static final Random rng = new Random();

    public static final float SQRT_2_OVER_PI = 0.7978845608f;
    public static final float INV_SQRT_2_OVER_PI = 0.3989422804f;
    public static final float GELU_APROX = 0.044715f;
    public static final float EPSILON5 = 1e-5f;

    private UtilMath() {}

    public static float round(float value, int x) 
    {
        float scale = (float) Math.pow(10, x);
        return Math.round(value * scale) / scale;    
    }
}
