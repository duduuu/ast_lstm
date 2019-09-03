# coding=utf-8
from __future__ import absolute_import

from io import StringIO

from collections import Iterable

import os
from pdfrw import PdfReader, PdfWriter, PdfName, PdfArray, PdfDict, PdfObject, IndirectPdfDict, PdfString
from pdfrw.objects.pdfname import BasePdfName

from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree
from asdl.transition_system import GenTokenAction, TransitionSystem, ApplyRuleAction, ReduceAction

from common.registerable import Registrable


def pdf_to_ast(grammar, x, tr):
    if len(tr) >= 300:
        raise NotImplementedError
    if isinstance(x, PdfDict):
        prod = grammar.get_prod_by_ctr_name('PdfDict')

        ast_nodes = []
        for y in x:
            prod_ = grammar.get_prod_by_ctr_name('Apply')

            pred_field = RealizedField(prod_['name'], value=str(y))

            if y in ['/Parent', '/P', '/Dest', '/Prev']:
                op_field = RealizedField(prod_['op'])

            else:
                tr.append(y)
                args_ast_node = pdf_to_ast(grammar, x[y], tr)
                del tr[len(tr) - 1]
                op_field = RealizedField(prod_['op'], value=args_ast_node)

            ast_node_ = AbstractSyntaxTree(prod_, [pred_field, op_field])
            ast_nodes.append(ast_node_)

        if x.stream:
            prod_ = grammar.get_prod_by_ctr_name('PdfString')

            var_field = RealizedField(prod_['value'], value=str(x.stream))

            ast_node = AbstractSyntaxTree(prod_, [var_field])

            ast_nodes.append(ast_node)

        arg_field = RealizedField(prod['args'], ast_nodes)

        ast_node = AbstractSyntaxTree(prod, [arg_field]) 

    elif isinstance(x, PdfObject):
        prod = grammar.get_prod_by_ctr_name('PdfObject')

        var_field = RealizedField(prod['value'], value=str(x))

        ast_node = AbstractSyntaxTree(prod, [var_field])

    elif isinstance(x, PdfArray):
        dict_nodes = []
        list_nodes = []
        for y in x:
            if isinstance(y, PdfDict):
                args_ast_node = pdf_to_ast(grammar, y, tr)
                dict_nodes.append(args_ast_node)

            elif isinstance(y, PdfArray):
                raise NotImplementedError

            elif isinstance(y, BasePdfName):
                list_nodes.append(str(y))

            elif isinstance(y, PdfObject):
                list_nodes.append(str(y))


        if dict_nodes:
            prod = grammar.get_prod_by_ctr_name('PdfArray')

            arg_field = RealizedField(prod['args'], dict_nodes)

            ast_node = AbstractSyntaxTree(prod, [arg_field]) 

        else:
            prod = grammar.get_prod_by_ctr_name('PdfList')

            var_field = RealizedField(prod['value'], value=tuple(list_nodes))

            ast_node = AbstractSyntaxTree(prod, [var_field])

    elif isinstance(x, PdfString):
        prod = grammar.get_prod_by_ctr_name('PdfString')

        var_field = RealizedField(prod['value'], value=str(x))

        ast_node = AbstractSyntaxTree(prod, [var_field])

    elif isinstance(x, BasePdfName):
        prod = grammar.get_prod_by_ctr_name('BasePdfName')

        var_field = RealizedField(prod['value'], value=str(x))

        ast_node = AbstractSyntaxTree(prod, [var_field])

    else:
        print(type(x))
        raise NotImplementedError

    return ast_node


def ast_to_pdf(asdl_ast):
    constructor_name = asdl_ast.production.constructor.name

    if constructor_name == 'PdfDict':
        pdict = {}
        for arg in asdl_ast['args'].value:
            # apply
            name = BasePdfName(asdl_ast['name'].value)
            if name in ['/Parent', '/P', '/Dest', '/Prev']:
                pass
            op = ast_to_pdf(asdl_ast['op'].value)

            pdict[name] = op

        x = PdfDict(pdict)

    elif constructor_name == 'PdfArray':
        parray = []
        for arg in asdl_ast['args'].value:
            args.append(ast_to_pdf(arg))

        x = PdfArray(pdfarray)

    elif constructor_name == 'PdfList':
        var = asdl_ast['value'].value

        x = PdfArray(list(var))

    elif constructor_name == 'PdfObject':
        var = asdl_ast['value'].value
        
        x = PdfObject(var)

    elif constructor_name == 'PdfStr':
        var = asdl_ast['value'].value
        
        x = PdfStr(var)

    elif constructor_name == 'BasePdfName':
        var = asdl_ast['value'].value
        
        x = BasePdfName(var)

    return x

def is_equal_ast(this_ast, other_ast):
    pass


@Registrable.register('pdf')
class PdfTransitionSystem(TransitionSystem):
    def tokenize_code(self, code, mode):
        return code.split(' ')

    def compare_ast(self, hyp_ast, ref_ast):
        return is_equal_ast(hyp_ast, ref_ast)

    def surface_code_to_ast(self, code):
        return pdf_to_ast(self.grammar, code)

    def ast_to_surface_code(self, asdl_ast):
        return ast_to_pdf(asdl_ast)

    def get_primitive_field_actions(self, realized_field):
        assert realized_field.cardinality == 'single'
        if realized_field.value is not None:
            return [GenTokenAction(realized_field.value)]
        else:
            return []


if __name__ == '__main__':
    path = 'pdfs'
    grammar = ASDLGrammar.from_text(open('pdf_asdl.txt').read())
    transition_system = PdfTransitionSystem(grammar)

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
    
        pdf = PdfReader(item_path)

        for page in pdf.pages:
            ast_tree = pdf_to_ast(grammar, page)
            ast_tree.sanity_check()
            print(ast_to_pdf(ast_tree))