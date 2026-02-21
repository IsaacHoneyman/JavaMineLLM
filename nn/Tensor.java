package nn;

import linalg.Matrix;

public class Tensor {
    // fields are public for performance overhead 
    
    public Matrix data;
    public Matrix grad; // gradients, (error signal)

    public Tensor[] parents;
    public Runnable backward;

    public Tensor(Matrix data){
        this.data = data;
        this.grad = new Matrix(data.rows, data.columns);
        this.parents = new Tensor[0];
        this.backward = () -> {};
    }

    public Tensor(Matrix data, Tensor[] parents, Runnable backward){
        this.data = data;
        this.grad = new Matrix(data.rows, data.columns);
        this.parents = parents.clone();
        this.backward = backward;
    }

    // --- Tensor Manipulation ---

    public static Tensor add(Tensor a, Tensor b) {
        Matrix outData = Matrix.add(a.data, b.data);
        Tensor out = new Tensor(outData);
        out.parents = new Tensor[] {a, b};
        
        out.backward = () -> {
            a.grad = Matrix.add(a.grad, out.grad);
            b.grad = Matrix.add(b.grad, out.grad);
        };

        return out;
    }

    public Tensor add(Tensor other){
        return add(this, other);
    }

    public static Tensor dot(Tensor a, Tensor b){
        Matrix outData = Matrix.dot(a.data, b.data);
        Tensor out = new Tensor(outData);
        out.parents = new Tensor[] {a, b};

        out.backward = () -> {
            a.grad = Matrix.add(a.grad, Matrix.dot(out.grad, b.data.transpose()));
            b.grad = Matrix.add(b.grad, Matrix.dot(a.data.transpose(), out.grad));
        };

        return out;
    }

    public Tensor dot(Tensor other){
        return dot(this, other);
    }

    public Tensor multiply(float scalar){
        Matrix outData = data.multiply(scalar);
        Tensor out = new Tensor(outData);
        out.parents = new Tensor[] {this};

        out.backward = () -> {
            this.grad = Matrix.add(this.grad, out.grad.multiply(scalar));
        };

        return out;
    }
}
