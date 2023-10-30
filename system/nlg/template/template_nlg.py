from convlab2.nlg.template.multiwoz import TemplateNLG

class WrappedTemplateNLG(TemplateNLG):
    def __init__(self, mode) -> None:
        super().__init__(is_user=False, mode=mode)