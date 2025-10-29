 🧮 Expression-to-Matrix Computation

The core idea of mapping any arithmetic expression (its abstract syntax tree, AST) into a chain of matrix operations is to treat the entire expression as a dataflow network — where variables and constants form an input vector, and a series of selection and transformation matrices route values through the network.
Each operation (addition, multiplication, logical operations, or even non-linear functions) corresponds to one or more matrix transformation steps.
Matrix multiplication, dot products, and other built-in vectorized operations then carry out the actual computation.

In essence, an expression is viewed as an arithmetic circuit or tensor network, where each node corresponds to a matrix or tensor merge operation that channels data from inputs to outputs.

Example: “ab + cd”

Take the expression a * b + c * d.
We can decompose its AST into the following computational steps:

Input Vector
Collect all variables into a single column vector:
v = [a, b, c, d]^T.

Step 1 – Select Operands
Construct a matrix M2 to route inputs to each multiplication’s left and right operands.
For example:

M2_left  = [[1,0,0,0],
            [0,0,1,0]]
M2_right = [[0,1,0,0],
            [0,0,0,1]]


Multiplying these with v yields:
left = [a, c]^T, right = [b, d]^T.

Step 2 – Elementwise Multiplication
Perform componentwise multiplication:
[a * b, c * d]^T.
This can be done using elementwise multiplication or the built-in dot function in WGSL:
dot([a, c], [b, d]) = a*b + c*d.
Alternatively, compute intermediate values p1 = a*b, p2 = c*d and form [p1, p2]^T.

Step 3 – Combine Results
Use another matrix to sum the results:
M3 = [[1, 1]], which acts on [p1, p2]^T to yield p1 + p2.

Here, M2 and M3 handle data routing and aggregation, while the actual arithmetic (*, +) is performed via scalar or vector operations in WGSL.
Thus, a complex expression becomes:

Input vector → (selection matrices → operations) → merge matrix

Generalization to Complex ASTs

This logic scales to any expression tree by traversing the AST in topological order (bottom-up):

Vectorization of Variables and Intermediates
Gather all leaf variables into a column vector.
Each new intermediate result (e.g. e = f + g) becomes a new “virtual variable,” appended to the vector.
As you compute nodes, the vector length grows.

Matrix Representation of Operations
For each operation node (add, subtract, multiply, function call, etc.), build one or more matrices to select its operands based on their positions in the current vector.
Binary operations use two selection rows/columns, unary ones use one.
The selected values are then processed by a scalar or vector operation.

Injecting Results
The node’s output is appended to the vector, ready for its parent node.
Once the entire AST is processed, one vector element holds the final result.

Example: h = (a + b) * (c - d)

Start with v = [a, b, c, d]^T.

Use matrices to select [a,b] → sum → p1, and [c,d] → difference → p2.

Append p1, p2 to the vector, then select [p1,p2] → multiply → h.
This process mirrors an AST evaluated bottom-up via linear selections + nonlinear node operations.

Nonlinear Functions and Conditional Branches
Nonlinear Functions

Matrix operations remain linear, but you can interleave nonlinear scalar functions (e.g., sin, log, exp) at each node:

Select operands with matrices.

Apply WGSL built-ins (sin(), exp(), etc.).

Continue matrix operations as usual.

Example: f = sin(a*b) + log(c)*d
Compute a*b (matrix-select and multiply) → apply sin().
Select c → apply log().
Then continue standard matrix assembly and addition.

Conditional Branches

Conditionals (if/else) can also be represented with matrices:

Encode condition values as 0/1 scalars.

Construct routing matrices that select based on the condition.

Example:

result = cond ? X : Y


Use:

M = [[cond, 0],
     [0, 1 - cond]]


Then result = M * [X, Y]^T.
This mimics logic gates — e.g., a Boolean AND gate can be represented by matrices such as
[[1,1,1,0],[0,0,0,1]], mapping {00,01,10,11} → {0,0,0,1}.
Thus, logical control flow can be embedded in linear algebra form.

WGSL Matrix Types and Operations

WGSL provides native matrix types and operators, ideal for this approach.

Examples:

let M4 : mat4x4f = mat4x4f(...);   // 4×4 matrix
let M4x3 : mat4x3f = mat4x3f(...); // 4×3 matrix
let C : mat4x3f = M4 * M4x3;       // Valid multiplication


Vector × matrix and matrix × vector are both supported.

Example:

vec3(9,8,7) * m2x3   // produces a 2-component vector
m2x3 * vec2(9,8)     // produces a 3-component vector


These operators allow WGSL shaders to directly execute the matrix transformation pipeline corresponding to the AST.

Summary

To convert an expression into matrix computation:

Traverse the AST bottom-up.

Map each operation into one or more selection and merge matrices.

Use WGSL scalar/vector/matrix operations to implement the actual arithmetic.

Chain all transformations linearly to form a GPU-executable matrix pipeline.

By doing this, complex expressions are computed efficiently using GPU-accelerated matrix/vector pipelines, fully compatible with WGSL’s native matrix semantics.

References

Tensor network and arithmetic circuit analogies: journals.aps.org

Matrix representation of Boolean logic: ozaner.github.io

WGSL matrix/vector operation examples: google.github.io