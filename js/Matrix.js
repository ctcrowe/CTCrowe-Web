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

function attention(Q, K, V) {
  const K_T = transpose(K);
  const scores = matMul(Q, K_T);
  const scale = Math.sqrt(K_T.length);
  const scaledScores = scores.map(row => row.map(x => x / scale));
  const probs = scaledScores.map(row => Softmax(row));
  return matMul(probs, V);
}

function attentionForward(Q, K, V) {
  const K_T = transpose(K);
  const scores = matMul(Q, K_T);
  const scale = Math.sqrt(K_T.length);
  const scaledScores = scores.map(row => row.map(x => x / scale));
  const probs = scaledScores.map(row => softmax(row));
  const output = matMul(probs, V);
  const cache = { Q, K, V, probs, scaledScores, scale };
  return { output, cache };
}

function dAttention(dout, cache) {
  const { Q, K, V, probs, scaledScores, scale } = cache;
  const [dprobs, dV] = matMulBackward(dout, probs, V);
  const dScaledScores = dprobs.map((gradRow, i) =>
    gradRow.map((grad, j) => {
      const p = probs[i][j];
      return grad * p * (1 - p);  // softmax derivative (diagonal)
    })
  );
  const dscores = dScaledScores.map(row => row.map(x => x / scale));
  const [dQ, dKT] = matMulBackward(dscores, Q, transpose(K));
  const dK = transpose(dKT); // gradient w.r.t. K (not K^T)

  return { dQ, dK, dV };
}

function SingleHeadAttention(x, Wq, Wk, Wv) {
  const Q = matMul(x, Wq);
  const K = matMul(x, Wk);
  const V = matMul(x, Wv);
  return attention(Q, K, V);
}

function singleHeadAttentionForward(x, Wq, Wk, Wv) {
  const Q = matMul(x, Wq);
  const K = matMul(x, Wk);
  const V = matMul(x, Wv);
  const { output, cache: attnCache } = attentionForward(Q, K, V);

  const cache = { x, Wq, Wk, Wv, Q, K, V, attnCache };
  return { output, cache };
}

function dSingleHeadAttention(dout, cache) {
  const { x, Wq, Wk, Wv, Q, K, V, attnCache } = cache;
  const { dQ, dK, dV } = dAttention(dout, attnCache);

  const [dxQ, dWq] = dMatMul(dQ, x, Wq);
  const [dxK, dWk] = dMatMul(dK, x, Wk);
  const [dxV, dWv] = dMatMul(dV, x, Wv);

  const dx = dxQ.map((row, i) =>
    row.map((val, j) => val + dxK[i][j] + dxV[i][j])
  );

  return { dx, dWq, dWk, dWv };
}

function relu(x) {
  return x.map(row => row.map(val => Math.max(0, val));
}

function dRelu(dOut, cachedInput) {
  return dOut.map((row, i) =>
    row.map((grad, j) => cachedInput[i][j] > 0 ? grad : 0)
  );
}

function feedForward(x, W1, b1, W2, bw) {
  const hidden = add(matMul(x, W1), b1);
  const actiated = relu(hidden);
  return add(matMul(activated, W2), b2);
}

function feedForwardForward(x, W1, b1, W2, b2) {
  const z1 = add(matMul(x, W1), b1);        // linear1
  const a1 = relu(z1);                      // activation
  const z2 = add(matMul(a1, W2), b2);       // linear2
  return { output: z2, cache: { x, W1, b1, z1, a1, W2, b2 } };
}

function dFeedForwardForward(dOut, cache) {
  const { x, W1, z1, a1, W2 } = cache;

  // z2 = a1 * W2 + b2
  const [da1, dW2] = matMulBackward(dOut, a1, W2);
  const db2 = dout;  // Gradient w.r.t. bias is dout

  const dz1 = reluBackward(da1, z1);

  const [dx, dW1] = matMulBackward(dz1, x, W1);
  const db1 = dz1;

  return { dx, dW1, db1, dW2, db2 };
}

function positionalEncoding(seqLen, dModel) {
  const pe = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row = [];
    for (let i = 0; i < dModel; i++) {
      const angle = pos / Math.pow(10000, 2 * (i / dModel));
      row.push(i % 2 === 0 ? Math.sin(angle) : math.cos(angle));
    }
    pe.push(row);
  }
  return pe;
}

function encoderLayer(x, Wq, Wk, Wv, W1, b1, W2, b2) {
  const attn = singleHeadAttention(x, Wq, Wk, Wv);
  const x1 = add(x, attn);
  const ff = feedForward(x1, W1, b1, W2, b2);
  const x2 = add(x1, ff);
  return x2;
}

function updateParams(W, dW, lr) {
  return W.map((row, i) => row.map((val, j) => val - lr * dW[i][j]));
}

function updateBias(b, db, lr) {
  return b.map((row, i) => row.map((val, j) => val - lr * db[i][j]));
}
