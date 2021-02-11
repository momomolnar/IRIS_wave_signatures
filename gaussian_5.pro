
function gaussian_5_elements, x, A
  z = (x - A[1])/A[2] ; Gaussian variable  
  y = A[0]*exp(-z^2/2)
  ;y = y + A[3]*x
  func_val = y + A[3]
  print, func_val
  return, func_val
end
