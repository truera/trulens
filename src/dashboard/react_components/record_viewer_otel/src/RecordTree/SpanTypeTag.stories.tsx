import type { Meta, StoryObj } from '@storybook/react';
import { SpanTypeTag } from './SpanTypeTag';

type Story = StoryObj<typeof SpanTypeTag>;

const meta: Meta<typeof SpanTypeTag> = {
  title: 'Components/RecordTree/SpanTypeTag',
  component: SpanTypeTag,
  args: {
    spanType: 'generation',
  },
};

export default meta;

export const Default: Story = {};

export const WithCustomStyling: Story = {
  args: {
    spanType: 'record_root',
    sx: {
      backgroundColor: 'primary.main',
      color: 'primary.contrastText',
    },
  },
};

export const DifferentSpanType: Story = {
  args: {
    spanType: 'record_root',
  },
};
