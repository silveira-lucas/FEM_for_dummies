from sympy.printing.numpy import NumPyPrinter

#%%

class MyPrinter(NumPyPrinter):
    
    def _print_Function(self, expr):
        '''
        This method overrides the original _print_Funciton from Sympy
        NumpyPrinter. With this method only the function attribue name is
        printed, the function variables are not printed. This is convenient
        when in the numerical code the function is actually an array and not
        a numerical function.

        Parameters
        ----------
        expr : sympy.Function
               sympy undefined function

        Returns
        -------
        expr.name : str
                    sympy.Function attribute name
        '''
        return '%s' % (expr.name )
    
    def _print_Derivative(self, expr):
        '''
        This method overrides the original _print_Derivative from Sympy 
        NumpyPrinter. This method adds the prefix '_dot', '_ddot', '_dddot', 
        ..., to the function name to be printed. This is convenient when in 
        the numerical code the function derivative is stored inside a 
        numpy.ndarray.
        
        Parameters
        ----------
        expr : sympy.function.Derivative
               n**{th} derivative of an undefined sympy function
        
        Returns
        -------
        a : str
            derivative array name
        '''
        
        # Get the derivative order
        dim = 'ot'
        for i in range(expr.derivative_count):
            dim = 'd' + dim
        
        # Get the derivative argument name
        a = expr.args[0].name

        # Check if the derivative name contains '[]'
        j = None
        for i, char in enumerate(a):
            if (char == '['):
                j = i
        if (j is not None):
            a = a[:j] + '_'+ dim + a[j:]
        else:
            a = a[:] + '_' + dim

        return a
    
    def _print_Integral(self, expr):
        '''
        This method translates the symbolic integral into a numerical
        trapezoidal integral, i.e. the numpy.trapz function.

        Parameters
        ----------
        expr : sympy expression

        Returns
        -------
        expr_num : str
                   numpy.trapz integral of the sympy expression
        '''
        
        part_1 = 'numpy.trapz'
        part_2 = MyPrinter().doprint(expr.args[0])
        part_3 = expr.args[1][0]
        
        expr_num = '%s(%s,%s)' % (part_1, part_2, part_3)
        return expr_num
    
    @staticmethod
    def print_comment(string, ident, max_columns=79):
        '''
        The function manipulates the string so it can be printed considering 
        the maximum of carachters per line defined taking into account the 
        identation of the funcitons.
        
        Parameters
        ----------
        string : string
                 python string to be formated
        ident : string
                python string containing the identation inside the function
        max_columns : integer, optional
                      maximum number of character per line. The default number
                      is 79, as recommended by the PEP8 stadard.
                      
        Returns
        -------
        comment : string
                  string formated to the maximum of line length desired
        '''
        words = string.split()
        
        comment = ''
        line_length = int(0)
        for word in words:    
            if (len(comment)==0):
                comment += ident + word
                line_length = len(ident) + len(word)
            elif (line_length + len(' ') + len(word) < max_columns):
                comment += ' ' + word
                line_length += len(' ') + len(word)            
            else:
                comment += ' ' + '\n' + ident + word
                line_length = len(ident) + len(word)
        return comment
