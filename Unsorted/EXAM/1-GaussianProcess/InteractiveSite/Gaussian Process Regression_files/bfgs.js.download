"use strict"

function BFGS(g, x0, max_iter) {

  var n = x0.length;
  max_iter = max_iter || 50;
  var a0 = 0.01;
  var bisection_maxiter = 10;

  var flags = [];

  function bisection_line_search(x0, p) {
    var tol = 1e-3;
    var a, a_lo = 0, a_hi = a0;
    var k = 0;
    for (k = 0; k < bisection_maxiter; ++k) {
      a = (a_hi + a_lo) / 2;
      var h = g(x0.add(p.scale(a))).dot(p);
      if (h > tol) a_hi = a;
      else if (h < tol) a_lo = a;
      else break;
    }
    return a;
  }

  function is_violated(xnew) {
    var violated = false;
    for (var i = 0; i < xnew.length; ++i) {
      if (xnew[i] < 1e-6) { violated = true; break; };
    }
    return violated;
  }

  var B = Float64Array.eye(n, n);
  var x = x0.copy();
  var k = 0;
  for (k = 0; k < max_iter; ++k) {
    var p = B.lu_solve(g(x).negate());
    var a = bisection_line_search(x, p);
    var xnew = x.add(p.scale(a));

    var violated = is_violated(xnew);
    var violations = 0;
    while (violated) {
      violations++;
      a = a / 2;
      xnew = x.add(p.scale(a));
      violated = is_violated(xnew);
      if (violations > 10) {
        return {optimum: x, iters: k, errors: ['violations exceeded']};
      }
    }

    var s = p.map(function(p_i, i) { return a * p_i; });
    var y = g(xnew).subtract(g(x));
    var first = Float64Array.outer(y, y).scale(1.0 / y.dot(s));
    var second = B.multiply(Float64Array.outer(s, s.transpose().multiply(B))).scale(1.0 / s.dot(B.multiply(s)));
    B = B.add(first).subtract(second);
    x = xnew.copy();
  }
  return {optimum: x, iters: k, errors: []};

};