import timeit

import numpy as np

print(timeit.timeit('np.linalg.norm(x)', setup='import numpy as np; x = np.arange(100)', number=1000))
print(timeit.timeit('np.sqrt(x.dot(x))', setup='import numpy as np; x = np.arange(100)', number=1000))
print(timeit.timeit('math.sqrt(np.sum(np.square(x)))', setup='import numpy as np;import math; x = np.arange(100)', number=1000))
print(timeit.timeit('math.sqrt(x.dot(x))', setup='import numpy as np;import math; x = np.arange(100)', number=1000))
print(timeit.timeit('4+np.dot(x,y)', setup='import numpy as np;import math; x = np.arange(100); y = np.arange(100)', number=1000))
print(timeit.timeit('4+x.dot(y)', setup='import numpy as np;import math; x = np.arange(100); y = np.arange(100)', number=1000))

x = np.array([1,2,3,4,5])
y = np.array([1,2,8,4,5])
print(np.dot(x,y))
print(x.dot(y))
print(x)
print(y)