while True:
    ans = input('pyLLE:\nBase Package or Package With pyLLEHelper? (B/H)\n')
    if ans == 'B':
        from ._llesolver import LLEsolver
        from ._analyzedisp import AnalyzeDisp
        break
    if ans == 'H':
        print("Importing base solver:")
        from ._llesolver import LLEsolver
        print("\nImporting base analyzer")
        from ._analyzedisp import AnalyzeDisp
        print("\nImporting pyLLEHelper")
        from .pyLLEHelper import pyLLEHelper
        break
    print("Please enter 'B' OR 'H'")

    #FIGURE OUT WHY THERE IS A . INFRONT OF THE IMPORTS. THEY ARE NOT HIDDEN FILES..?
