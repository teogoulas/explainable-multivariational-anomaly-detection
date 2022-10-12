class ResourceError(ValueError):
    """Exception raised for errors during the AWS resource manipulation

    Attributes:
        error_code: Exception error code
        operation_name: Operation name during which the exception occurred
        message: Explanation of the error
    """

    def __init__(self, error_code: str, operation_name: str, message: str):
        self.MSG_TEMPLATE = 'An error occurred ({error_code}) when calling the {operation_name} operation: {' \
                            'error_message} '
        self.error_code = error_code
        self.operation_name = operation_name
        self.message = message
        self.response = {
            'Error': {
                'Message': message,
                'Code': error_code
            },
            'Message': self.MSG_TEMPLATE.format(error_code=self.error_code, operation_name=self.operation_name,
                                                error_message=self.message)
        }
        super().__init__(self.message)
