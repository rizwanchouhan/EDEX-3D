import ast

def generate_comments(filename):
    with open(filename, 'r') as file:
        source_code = file.read()

    tree = ast.parse(source_code)

    comments = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            comments.append(f"# Function: {node.name}")
            for arg in node.args.args:
                comments.append(f"#   - Parameter: {arg.arg}")
            if node.returns:
                comments.append(f"#   - Returns: {node.returns}")
        elif isinstance(node, ast.ClassDef):
            comments.append(f"# Class: {node.name}")
            for base in node.bases:
                comments.append(f"#   - Base class: {base.id}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    comments.append(f"#   - Method: {item.name}")
                    for arg in item.args.args:
                        comments.append(f"#     - Parameter: {arg.arg}")
                    if item.returns:
                        comments.append(f"#     - Returns: {item.returns}")
    
    return '\n'.join(comments)

def add_comments_to_file(input_file, output_file):
    generated_comments = generate_comments(input_file)

    with open(output_file, 'w') as file:
        with open(input_file, 'r') as input_file_content:
            file.write(generated_comments)
            file.write('\n\n')
            file.write(input_file_content.read())

# Example usage:
input_filename = 'graph.py'
output_filename = 'output.py'
add_comments_to_file(input_filename, output_filename)
