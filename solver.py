import cv2
import numpy as np

allowedBitFields = [0, 1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7, 1 << 8]

def arraySum(arrayz):
      i = 0
      for x in arrayz:
            i += x
      return i

allAllowed = arraySum(allowedBitFields)

def applyAllowedValuesMask(board, allowedValues, x, y):
      mask = ~allowedBitFields[board[x][y]];

      for maskApplyX in range(0,9):
            allowedValues[maskApplyX][y] &= mask

      allowedValuesRow = allowedValues[x]

      for maskApplyY in range(0,9):
            allowedValuesRow[maskApplyY] &= mask

      sectionX1 = sectionX2 = 0
      if x == 0:
          sectionX1 = x + 1
          sectionX2 = x + 2
      elif x == 1:
          sectionX1 = x - 1
          sectionX2 = x + 1
      elif x == 2:
          sectionX1 = x - 2
          sectionX2 = x - 1
      elif x == 3:
          sectionX1 = x + 1
          sectionX2 = x + 2
      elif x == 4:
          sectionX1 = x - 1
          sectionX2 = x + 1
      elif x == 5:
          sectionX1 = x - 2
          sectionX2 = x - 1
      elif x == 6:
          sectionX1 = x + 1
          sectionX2 = x + 2
      elif x == 7:
          sectionX1 = x - 1
          sectionX2 = x + 1
      elif x == 8:
          sectionX1 = x - 2
          sectionX2 = x - 1

      sectionY1 = sectionY2 = 0;
      if y == 0:
          sectionY1 = y + 1
          sectionY2 = y + 2
      elif y == 1:
          sectionY1 = y - 1
          sectionY2 = y + 1
      elif y == 2:
          sectionY1 = y - 2
          sectionY2 = y - 1
      elif y == 3:
          sectionY1 = y + 1
          sectionY2 = y + 2
      elif y == 4:
          sectionY1 = y - 1
          sectionY2 = y + 1
      elif y == 5:
          sectionY1 = y - 2
          sectionY2 = y - 1
      elif y == 6:
          sectionY1 = y + 1
          sectionY2 = y + 2
      elif y == 7:
          sectionY1 = y - 1
          sectionY2 = y + 1
      elif y == 8:
          sectionY1 = y - 2
          sectionY2 = y - 1

      allowedValuesRow1 = allowedValues[sectionX1]
      allowedValuesRow2 = allowedValues[sectionX2]

      allowedValuesRow1[sectionY1] &= mask
      allowedValuesRow1[sectionY2] &= mask
      allowedValuesRow2[sectionY1] &= mask
      allowedValuesRow2[sectionY2] &= mask

def printBoard(board):
      for x in range(0,9):
            for y in range(0,9):
                  print(board[x][y], end=' ')
            print('')

def copyGameMatrix(matrix):
      return matrix.copy()

def countSetBits(value):
      count = 0
      while value > 0:
            value = value & (value - 1)
            count += 1
      return count

def getLastSetBitIndex(value):
      bitIndex = 0
      while value > 0:
            bitIndex+=1
            value >>= 1
      return bitIndex

def setValue(board, allowedValues, value, x, y):
      board[x][y] = value
      allowedValues[x][y] = 0
      applyAllowedValuesMask(board, allowedValues, x, y)

def applyLineCandidateConstraints(allowedValues):
      for value in range(1,10):
            sectionAvailabilityColumn = [0,0,0,0,0,0,0,0,0]
            valueMask = allowedBitFields[value]
            valueRemoveMask = ~valueMask
            
            for x in range(0,9):
                  finalX = x
                  for y in range(0,9):
                        if (allowedValues[finalX][y] & valueMask) != 0:
                              sectionAvailabilityColumn[finalX] |= (1 << int(y / 3))
                
                  if finalX == 2 or finalX == 5 or finalX == 8:
                        for scanningX in range(finalX - 2,finalX+1):
                              bitCount = countSetBits(sectionAvailabilityColumn[scanningX])
                              if bitCount == 1:
                                    for applyX in range(finalX - 2, finalX+1):
                                          if scanningX != applyX:
                                                for applySectionY in range(0,3):
                                                      if (sectionAvailabilityColumn[scanningX] & (1 << applySectionY)) != 0:
                                                            for applyY in range(applySectionY * 3,(applySectionY + 1) * 3):
                                                                  allowedValues[applyX][applyY] &= valueRemoveMask
                              if bitCount == 2 and scanningX < finalX:
                                    for scanningSecondPairX in range(scanningX + 1,finalX+1):
                                          if sectionAvailabilityColumn[scanningX] == sectionAvailabilityColumn[scanningSecondPairX]:
                                                applyX = 0
                                                if scanningSecondPairX != finalX:
                                                      applyX = finalX
                                                elif scanningSecondPairX - scanningX > 1:
                                                      applyX = scanningSecondPairX - 1
                                                else:
                                                      applyX = scanningX - 1
                                                for applySectionY in range(0,3):
                                                      if (sectionAvailabilityColumn[scanningX] & (1 << applySectionY)) != 0:
                                                            for applyY in range(applySectionY * 3, (applySectionY + 1) * 3):
                                                                  allowedValues[applyX][applyY] &= valueRemoveMask
                                                break;
            
            sectionAvailabilityRow = [0,0,0,0,0,0,0,0,0]
            for y in range(0,9):
                  finalY = y
                  for x in range(0,9):
                        if (allowedValues[x][finalY] & valueMask) != 0:
                              sectionAvailabilityRow[finalY] |= (1 << int(x / 3))
                  if finalY == 2 or finalY == 5 or finalY == 8:
                        for scanningY in range(finalY - 2,finalY+1):
                              bitCount = countSetBits(sectionAvailabilityRow[scanningY])
                              if bitCount == 1:
                                    for applyY in range(finalY - 2,finalY+1):
                                          if scanningY != applyY:
                                                for applySectionX in range(0,3):
                                                      if (sectionAvailabilityRow[scanningY] & (1 << applySectionX)) != 0:
                                                            for applyX in range(applySectionX * 3,(applySectionX + 1) * 3):
                                                                  allowedValues[applyX][applyY] &= valueRemoveMask
                              if bitCount == 2 and scanningY < finalY:
                                    for scanningSecondPairY in range(scanningY + 1,finalY+1):
                                          if sectionAvailabilityRow[scanningY] == sectionAvailabilityRow[scanningSecondPairY]:
                                                applyY=0
                                                if scanningSecondPairY != finalY:
                                                      applyY = finalY
                                                elif scanningSecondPairY - scanningY > 1:
                                                      applyY = scanningSecondPairY - 1
                                                else:
                                                      applyY = scanningY - 1
                                                for applySectionX in range(0,3):
                                                      if (sectionAvailabilityRow[scanningY] & (1 << applySectionX)) != 0:
                                                            for applyX in range(applySectionX * 3,(applySectionX + 1) * 3):
                                                                  allowedValues[applyX][applyY] &= valueRemoveMask

def applyNakedPairs(allowedValues):
      for x in range(0,9):
            for y in range(0,9):
                  value = allowedValues[x][y]
                  if countSetBits(value) == 2:
                        for scanningY in range(y + 1, 9):
                              if allowedValues[x][scanningY] == value:
                                    removeMask = ~value
                                    for applyY in range(0,9):
                                          if applyY != y and applyY != scanningY:
                                                allowedValues[x][applyY] &= removeMask
        
      for y in range(0,9):
            for x in range(0,9):
                  value = allowedValues[x][y]
                  if value != 0 and countSetBits(value) == 2:
                        for scanningX in range(x + 1, 9):
                              if allowedValues[scanningX][y] == value:
                                    removeMask = ~value
                                    for applyX in range(0,9):
                                          if applyX != x and applyX != scanningX:
                                                allowedValues[applyX][y] &= removeMask;

def moveNothingElseAllowed(board, allowedValues):
      moveCount = 0
      for x in range(0,9):
            allowedValuesRow = allowedValues[x]
            for y in range(0,9):
                  currentAllowedValues = allowedValuesRow[y]
                  if countSetBits(currentAllowedValues) == 1:
                        setValue(board, allowedValues, getLastSetBitIndex(currentAllowedValues), x, y)
                        moveCount+=1
      return moveCount

def moveNoOtherRowOrColumnAllowed(board, allowedValues):
      moveCount = 0
      for value in range(1,10):
            allowedBitField = allowedBitFields[value]
            for x in range(0,9):
                  allowedY = -1
                  allowedValuesRow = allowedValues[x]
                  for y in range(0,9):
                        if (allowedValuesRow[y] & allowedBitField) > 0:
                              if allowedY < 0:
                                    allowedY = y
                              else:
                                    allowedY = -1
                                    break
          
                  if allowedY >= 0:
                        setValue(board, allowedValues, value, x, allowedY)
                        moveCount+=1

            for y in range(0,9):
                  allowedX = -1
                  for x in range(0,9):
                        if (allowedValues[x][y] & allowedBitField) > 0:
                              if allowedX < 0:
                                    allowedX = x
                              else:
                                    allowedX = -1
                                    break
          
                  if allowedX >= 0:
                        setValue(board, allowedValues, value, allowedX, y)
                        moveCount+=1
      return moveCount

def attemptBruteForce(board, allowedValues, placedNumberCount):
      for x in range(0,9):
            allowedValuesRow = allowedValues[x]
            boardRow = board[x]
            for y in range(0,9):
                  if boardRow[y] == 0:
                        for value in range(1,10):
                              if (allowedValuesRow[y] & allowedBitFields[value]) > 0:
                                    testBoard = copyGameMatrix(board)
                                    testAllowedValues = copyGameMatrix(allowedValues)
                                    setValue(testBoard, testAllowedValues, value, x, y)
                                    placedNumbers = solveBoard(testBoard, testAllowedValues, placedNumberCount + 1)
                                    if placedNumbers == 81:
                                          return testBoard
                        return None
      return None

def solveBoard(board, allowedValues, placedNumberCount):
      lastPlacedNumbersCount = 0
      while placedNumberCount - lastPlacedNumbersCount > 3 and placedNumberCount < 68 and placedNumberCount > 10:
            lastPlacedNumbersCount = placedNumberCount
            placedNumberCount += moveNothingElseAllowed(board, allowedValues)
            placedNumberCount += moveNoOtherRowOrColumnAllowed(board, allowedValues)
            placedNumberCount += moveNothingElseAllowed(board, allowedValues)
            if placedNumberCount < 35:
                  applyNakedPairs(allowedValues)
                  applyLineCandidateConstraints(allowedValues)

      if placedNumberCount < 81:
            bruteForcedBoard = attemptBruteForce(board, allowedValues, placedNumberCount)
            if bruteForcedBoard != None:
                  placedNumberCount = 0
                  for x in range(0,9):
                        for y in range(0,9):
                              board[x][y] = bruteForcedBoard[x][y]
                              if bruteForcedBoard[x][y] > 0:
                                    placedNumberCount+=1
      return placedNumberCount

def solve(board):
      allowedValues=[]
      placedNumberCount = 0;
      for x in range(0,9):
            rows = []
            for y in range(0,9):
                  rows.append(allAllowed)
            allowedValues.append(rows)
      
      for x in range(0,9):
            for y in range(0,9):
                  if board[x][y] > 0:
                        allowedValues[x][y] = 0
                        applyAllowedValuesMask(board, allowedValues, x, y)
                        placedNumberCount+=1
      return solveBoard(board, allowedValues, placedNumberCount);

def boardFromLabels(labels):
      board=[]
      i = 0
      for x in range(0,9):
            rows = []
            for y in range(0,9):
                  rows.append(int(labels[i]))
                  i+=1
            board.append(rows)
      return board


#main

##filename = "3.jpg"
##frame = cv2.imread(filename)
##board,labels,centers = rec.recognizer(frame,filename)

##convert to numpy array of 9x9
##grid = np.array(labels).reshape(9,9)

##find the indices of empty cells
##gz_indices = zip(*np.where(grid==0))

##center co-ordinates of all the cells
##gz_centers = np.array(centers).reshape(9,9,2)

##if solve(board) == 81:
##      print("Done")
##      printBoard(board)
##      grid = board
##      rec.trace(grid,gz_indices,gz_centers)
      
##      for row,col in gz_indices:
##          cv2.putText(warp,str(grid[row][col]),tuple(gz_centers[row][col]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
##      cv2.imshow("Solved",warp)
##else:
##      print("Error")
