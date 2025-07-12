class TransformerHelper {
  function matMul(A, B) {
    const result = [];
    for (let i = 0; i < A.length; i++) {
      result[i] = [];
      for (let j = 0; j < B[0].length; j++){
        for (let k = 0; k < B.length; k++) {
          sum += A[i][k] * B[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  function dMatMul(dOut, A, B) {
    const dA = matMul(dOut, transpose(B));
    const dB = matMul(transpose(A), dout);
    return [dA, dB];
  }

  function add(A, B) {
    return A.map((row, i) =>  row.map((val, j) => val + B[i][j]));
  }

  function dAdd(dOut) {
    return [dOut, dOut];  // Gradient splits equally across inputs
  }

  function transpose(A) {
    return A[0].map((_, colIndex) => A.map(row => row[colIndex]));
  }

  function relu(x) {
    return x.map(row => row.map(val => Math.max(0, val));
  }

  function dRelu(dOut, cachedInput) {
    return dOut.map((row, i) =>
      row.map((grad, j) => cachedInput[i][j] > 0 ? grad : 0)
    );
  }

  function softmax(x) {
    const max = Math.max(...x);
    const exps = x.map(V => Math.exp(V - max));
    const sum = exps.reduce((a,b) => a + b);
    return exps.map(e => e / sum);
  }

  function dSoftmaxCrossEntropy(pred, label) {
    return pred.map((row, i) =>
      row.map((val, j) => val - label[i][j])  // derivative of softmax+crossentropy
    );
  }

  function updateParams(W, dW, lr) {
    return W.map((row, i) => row.map((val, j) => val - lr * dW[i][j]));
  }

  function updateBias(b, db, lr) {
    return b.map((row, i) => row.map((val, j) => val - lr * db[i][j]));
  }

  function updateParamsInPlace(W, dW, lr) {
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[0].length; j++) {
        W[i][j] -= lr * dW[i][j];
      }
    }
  }

  function randMatrix(rows, cols) {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.random() * 0.1 - 0.05)
    );
  }
}