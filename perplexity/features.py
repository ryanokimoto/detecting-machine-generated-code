class CodeFeatures:
    line_count: int = 0
    char_count: int = 0
    avg_line_length: float = 0.0
    max_line_length: int = 0
    indent_depth_avg: float = 0.0
    indent_depth_max: int = 0
    
    num_tokens: int = 0
    unique_tokens: int = 0
    token_entropy: float = 0.0
    
    num_identifiers: int = 0
    avg_identifier_length: float = 0.0
    identifier_entropy: float = 0.0
    snake_case_ratio: float = 0.0
    camel_case_ratio: float = 0.0
    
    comment_ratio: float = 0.0
    num_comments: int = 0
    
    num_functions: int = 0
    num_loops: int = 0
    num_conditionals: int = 0
    nesting_depth_max: int = 0
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.line_count,
            self.char_count,
            self.avg_line_length,
            self.max_line_length,
            self.indent_depth_avg,
            self.indent_depth_max,
            self.num_tokens,
            self.unique_tokens,
            self.token_entropy,
            self.num_identifiers,
            self.avg_identifier_length,
            self.identifier_entropy,
            self.snake_case_ratio,
            self.camel_case_ratio,
            self.comment_ratio,
        ], dtype=np.float32)

